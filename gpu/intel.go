package gpu

import (
	"encoding/binary"
	"fmt"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
)

// Intel GPUs don't expose a busy percentage in sysfs. Their drivers export
// engine busyness through a perf PMU instead:
//
//   - i915: /sys/bus/event_source/devices/i915 (integrated) or
//     i915_<pci address> (discrete), with one "<engine>-busy" event per
//     engine counting busy nanoseconds. Opening these events doesn't wake the
//     GPU, so they stay open and the rate is computed between two calls of
//     GetDevices. The first call returns no utilization.
//   - xe: /sys/bus/event_source/devices/xe_<pci address>, with the
//     "engine-active-ticks" and "engine-total-ticks" events per GT, engine
//     class and instance (GuC interface 1.14.1 or newer). An open event holds
//     forcewake and a runtime PM reference, which keeps the GPU out of power
//     saving states, so the events are only open for a short sample window
//     per call.
//
// Utilization is the busiest engine class, averaged over its instances. Using
// all engine classes instead of only render/compute makes hardware video
// decoding and encoding (e.g. camera streams in Frigate) visible.
//
// See https://docs.kernel.org/gpu/i915.html#perf and
// https://docs.kernel.org/gpu/xe/xe_pmu.html.

const (
	// Sample window of the xe engine activity counters per call.
	xeSampleWindow = 250 * time.Millisecond

	// DRM_XE_ENGINE_CLASS_RENDER ... DRM_XE_ENGINE_CLASS_COMPUTE
	xeEngineClasses = 5
	// Upper bound of engine instances per class to probe. Engine instances
	// of xe are logical and therefore contiguous.
	xeMaxEngineInstances = 16

	// Format of the xe PMU config, see /sys/bus/event_source/devices/xe_*/format
	xeConfigGTShift       = 60
	xeConfigClassShift    = 20
	xeConfigInstanceShift = 12
)

// perfCounter is an open perf event counting since it was opened.
type perfCounter interface {
	Read() (uint64, error)
	Close() error
}

// perfOpenFunc opens a system wide perf event of the PMU type on the cpu.
type perfOpenFunc func(pmuType uint32, cpu int, config uint64) (perfCounter, error)

type perfEvent int

func openPerfEvent(pmuType uint32, cpu int, config uint64) (perfCounter, error) {
	attr := unix.PerfEventAttr{
		Type:   pmuType,
		Config: config,
	}
	attr.Size = uint32(unsafe.Sizeof(attr)) //nolint:gosec // size of the struct for the kernel
	fd, err := unix.PerfEventOpen(&attr, -1, cpu, -1, unix.PERF_FLAG_FD_CLOEXEC)
	if err != nil {
		return nil, err
	}
	return perfEvent(fd), nil
}

func (e perfEvent) Read() (uint64, error) {
	var buf [8]byte
	n, err := unix.Read(int(e), buf[:])
	if err != nil {
		return 0, err
	}
	if n != len(buf) {
		return 0, fmt.Errorf("short read of %d bytes", n)
	}
	return binary.NativeEndian.Uint64(buf[:]), nil
}

func (e perfEvent) Close() error {
	return unix.Close(int(e))
}

// pmu is a perf event source in /sys/bus/event_source/devices.
type pmu struct {
	dir     string
	pmuType uint32
	cpu     int
}

func (c *collector) readPMU(name string) (pmu, bool) {
	dir := filepath.Join(c.sysRoot, "bus", "event_source", "devices", name)
	pmuType, err := strconv.ParseUint(readString(filepath.Join(dir, "type")), 10, 32)
	if err != nil {
		return pmu{}, false
	}
	// Uncore PMUs take events on one CPU only, e.g. "0" or "0-3"
	cpu := 0
	cpumask := readString(filepath.Join(dir, "cpumask"))
	if end := strings.IndexAny(cpumask, ",-"); end >= 0 {
		cpumask = cpumask[:end]
	}
	if value, err := strconv.Atoi(cpumask); err == nil {
		cpu = value
	}
	return pmu{dir: dir, pmuType: uint32(pmuType), cpu: cpu}, true
}

// eventConfig parses an event like "config=0x1000" (i915) or
// "event=0x02" (xe, bits 0-11 of config).
func (p pmu) eventConfig(name string) (uint64, bool) {
	value := readString(filepath.Join(p.dir, "events", name))
	key, number, found := strings.Cut(value, "=")
	if !found || (key != "config" && key != "event") {
		return 0, false
	}
	config, err := strconv.ParseUint(number, 0, 64)
	if err != nil {
		return 0, false
	}
	return config, true
}

// pmuName returns the name of the PMU of a device, colons are reserved by
// tools/perf.
func pmuName(prefix, address string) string {
	return prefix + "_" + strings.ReplaceAll(address, ":", "_")
}

type engineCounter struct {
	class   string
	counter perfCounter
}

// i915Counters are the open busy counters of an i915 device.
type i915Counters struct {
	engines []engineCounter
	busy    []uint64
	time    time.Time
}

func (s *i915Counters) close() {
	for _, engine := range s.engines {
		_ = engine.counter.Close()
	}
}

func (c *collector) i915Utilization(address string) *float64 {
	c.mu.Lock()
	defer c.mu.Unlock()

	counters := c.i915[address]
	if counters == nil {
		counters = c.openI915(address)
		if counters != nil {
			if c.i915 == nil {
				c.i915 = map[string]*i915Counters{}
			}
			c.i915[address] = counters
		}
		// Utilization needs a second reading
		return nil
	}

	busy, err := readCounters(counters.engines)
	now := c.now()
	if err != nil {
		counters.close()
		delete(c.i915, address)
		return nil
	}
	elapsed := now.Sub(counters.time)
	previous := counters.busy
	counters.busy, counters.time = busy, now
	if elapsed <= 0 {
		return nil
	}

	classes := make([]string, len(busy))
	ratios := make([]float64, len(busy))
	for i, engine := range counters.engines {
		classes[i] = engine.class
		if busy[i] > previous[i] {
			ratios[i] = float64(busy[i]-previous[i]) / float64(elapsed.Nanoseconds())
		}
	}
	return busiestClass(classes, ratios)
}

func (c *collector) openI915(address string) *i915Counters {
	// Integrated GPUs, which are on the root bus, have a PMU without address.
	names := []string{pmuName("i915", address)}
	if strings.HasPrefix(address, "0000:00:") {
		names = append(names, "i915")
	}

	for _, name := range names {
		p, ok := c.readPMU(name)
		if !ok {
			continue
		}
		events, _ := filepath.Glob(filepath.Join(p.dir, "events", "*-busy"))
		slices.Sort(events)

		counters := &i915Counters{}
		for _, event := range events {
			event = filepath.Base(event)
			config, ok := p.eventConfig(event)
			if !ok {
				continue
			}
			counter, err := c.perfOpen(p.pmuType, p.cpu, config)
			if err != nil {
				// e.g. missing CAP_PERFMON
				continue
			}
			// "vcs1-busy" belongs to the class "vcs"
			class := strings.TrimRight(strings.TrimSuffix(event, "-busy"), "0123456789")
			counters.engines = append(counters.engines, engineCounter{class: class, counter: counter})
		}
		if len(counters.engines) == 0 {
			return nil
		}

		busy, err := readCounters(counters.engines)
		if err != nil {
			counters.close()
			return nil
		}
		counters.busy, counters.time = busy, c.now()
		return counters
	}
	return nil
}

// pruneI915 closes the counters of devices which are gone.
func (c *collector) pruneI915(present []string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for address, counters := range c.i915 {
		if !slices.Contains(present, address) {
			counters.close()
			delete(c.i915, address)
		}
	}
}

type xeEngine struct {
	class         string
	active, total perfCounter
}

func (c *collector) xeUtilization(devicePath, address string) *float64 {
	// Don't wake up a runtime suspended GPU, it is idle
	if readString(filepath.Join(devicePath, "power", "runtime_status")) == "suspended" {
		idle := 0.0
		return &idle
	}

	p, ok := c.readPMU(pmuName("xe", address))
	if !ok {
		return nil
	}
	activeEvent, ok := p.eventConfig("engine-active-ticks")
	if !ok {
		return nil
	}
	totalEvent, ok := p.eventConfig("engine-total-ticks")
	if !ok {
		return nil
	}

	var engines []xeEngine
	defer func() {
		for _, engine := range engines {
			_ = engine.active.Close()
			_ = engine.total.Close()
		}
	}()

	for _, gt := range xeGTs(devicePath) {
		for class := range uint64(xeEngineClasses) {
			for instance := range uint64(xeMaxEngineInstances) {
				config := gt<<xeConfigGTShift | class<<xeConfigClassShift | instance<<xeConfigInstanceShift
				active, err := c.perfOpen(p.pmuType, p.cpu, activeEvent|config)
				if err != nil {
					// No further instance of this class on the GT
					break
				}
				total, err := c.perfOpen(p.pmuType, p.cpu, totalEvent|config)
				if err != nil {
					_ = active.Close()
					break
				}
				engines = append(engines, xeEngine{
					class:  strconv.FormatUint(class, 10),
					active: active,
					total:  total,
				})
			}
		}
	}
	if len(engines) == 0 {
		return nil
	}

	first, err := readXeCounters(engines)
	if err != nil {
		return nil
	}
	c.sleep(xeSampleWindow)
	second, err := readXeCounters(engines)
	if err != nil {
		return nil
	}

	var classes []string
	var ratios []float64
	for i, engine := range engines {
		if second[i][1] <= first[i][1] || second[i][0] < first[i][0] {
			continue
		}
		active := second[i][0] - first[i][0]
		total := second[i][1] - first[i][1]
		classes = append(classes, engine.class)
		ratios = append(ratios, float64(active)/float64(total))
	}
	return busiestClass(classes, ratios)
}

// xeGTs returns the IDs of the GTs of all tiles.
func xeGTs(devicePath string) []uint64 {
	matches, _ := filepath.Glob(filepath.Join(devicePath, "tile*", "gt*"))
	var gts []uint64
	for _, match := range matches {
		id, err := strconv.ParseUint(strings.TrimPrefix(filepath.Base(match), "gt"), 10, 8)
		if err == nil && !slices.Contains(gts, id) {
			gts = append(gts, id)
		}
	}
	if len(gts) == 0 {
		return []uint64{0}
	}
	slices.Sort(gts)
	return gts
}

func readCounters(engines []engineCounter) ([]uint64, error) {
	values := make([]uint64, len(engines))
	for i, engine := range engines {
		value, err := engine.counter.Read()
		if err != nil {
			return nil, err
		}
		values[i] = value
	}
	return values, nil
}

func readXeCounters(engines []xeEngine) ([][2]uint64, error) {
	values := make([][2]uint64, len(engines))
	for i, engine := range engines {
		active, err := engine.active.Read()
		if err != nil {
			return nil, err
		}
		total, err := engine.total.Read()
		if err != nil {
			return nil, err
		}
		values[i] = [2]uint64{active, total}
	}
	return values, nil
}

// busiestClass averages the busy ratios per engine class and returns the
// highest one in percent.
func busiestClass(classes []string, ratios []float64) *float64 {
	if len(ratios) == 0 {
		return nil
	}
	sums := map[string]float64{}
	counts := map[string]int{}
	for i, class := range classes {
		sums[class] += min(max(ratios[i], 0), 1)
		counts[class]++
	}
	busiest := 0.0
	for class, sum := range sums {
		busiest = max(busiest, sum/float64(counts[class]))
	}
	busiest *= 100
	return &busiest
}
