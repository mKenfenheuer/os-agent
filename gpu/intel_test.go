package gpu

import (
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

type perfKey struct {
	pmuType uint32
	config  uint64
}

// fakePerf returns the next of the values of an event on every read, the
// last one repeats.
type fakePerf struct {
	mu      sync.Mutex
	values  map[perfKey][]uint64
	reads   map[perfKey]int
	open    int
	readErr error
}

type fakeCounter struct {
	perf   *fakePerf
	key    perfKey
	closed bool
}

func newFakePerf(values map[perfKey][]uint64) *fakePerf {
	return &fakePerf{values: values, reads: map[perfKey]int{}}
}

func (f *fakePerf) openEvent(pmuType uint32, _ int, config uint64) (perfCounter, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	key := perfKey{pmuType, config}
	if _, ok := f.values[key]; !ok {
		return nil, unix.ENOENT
	}
	f.open++
	return &fakeCounter{perf: f, key: key}, nil
}

func (f *fakePerf) openCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.open
}

func (c *fakeCounter) Read() (uint64, error) {
	f := c.perf
	f.mu.Lock()
	defer f.mu.Unlock()
	if c.closed {
		return 0, unix.EBADF
	}
	if f.readErr != nil {
		return 0, f.readErr
	}
	values := f.values[c.key]
	i := min(f.reads[c.key], len(values)-1)
	f.reads[c.key]++
	return values[i], nil
}

func (c *fakeCounter) Close() error {
	f := c.perf
	f.mu.Lock()
	defer f.mu.Unlock()
	if c.closed {
		return unix.EBADF
	}
	c.closed = true
	f.open--
	return nil
}

// fakeClock advances by two seconds on every call of devices.
type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func (f *fakeClock) Now() time.Time {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.now
}

func (f *fakeClock) devices(c *collector) []Device {
	f.mu.Lock()
	f.now = f.now.Add(2 * time.Second)
	f.mu.Unlock()
	return c.devices()
}

func addPMU(t *testing.T, sysRoot, name, pmuType string, events map[string]string) {
	t.Helper()
	dir := filepath.Join(sysRoot, "bus", "event_source", "devices", name)
	writeFile(t, filepath.Join(dir, "type"), pmuType+"\n")
	writeFile(t, filepath.Join(dir, "cpumask"), "0\n")
	for event, config := range events {
		writeFile(t, filepath.Join(dir, "events", event), config+"\n")
	}
}

func i915Engine(class, instance uint64) uint64 {
	// __I915_PMU_ENGINE(class, instance, I915_SAMPLE_BUSY)
	return class<<12 | instance<<4
}

func xeConfig(event, gt, class, instance uint64) uint64 {
	return event | gt<<60 | class<<20 | instance<<12
}

func setupI915(t *testing.T) (*collector, *fakePerf, *fakeClock) {
	t.Helper()
	c := newTestCollector(t, "", nil)
	addCard(t, c.sysRoot, "card0", "0000:00:02.0", "i915", map[string]string{
		"vendor": "0x8086\n",
		"device": "0x46a6\n",
	})
	addPMU(t, c.sysRoot, "i915", "18", map[string]string{
		"rcs0-busy":        "config=0x0",
		"rcs0-busy.unit":   "ns",
		"bcs0-busy":        "config=0x1000",
		"vcs0-busy":        "config=0x2000",
		"vcs1-busy":        "config=0x2010",
		"vcs0-wait":        "config=0x2001",
		"actual-frequency": "config=0x100000",
	})
	// Two seconds between readings
	perf := newFakePerf(map[perfKey][]uint64{
		{18, i915Engine(0, 0)}: {0, 1e9, 3.5e9},
		{18, i915Engine(1, 0)}: {0, 1.5e9, 1.5e9},
		{18, i915Engine(2, 0)}: {0, 2e9, 2e9},
		{18, i915Engine(2, 1)}: {0, 1.2e9, 1.2e9},
		{18, 0x2001}:           {0},
		{18, 0x100000}:         {0},
	})
	clock := &fakeClock{now: time.Unix(1000, 0)}
	c.perfOpen = perf.openEvent
	c.now = clock.Now
	return c, perf, clock
}

func TestI915Utilization(t *testing.T) {
	c, perf, clock := setupI915(t)

	devices := clock.devices(c)
	if len(devices) != 1 || devices[0].Utilization != nil {
		t.Fatalf("first call must not return utilization: %+v", devices)
	}
	if perf.openCount() != 4 {
		t.Errorf("expected 4 open busy counters, got %d", perf.openCount())
	}

	// rcs 50 %, bcs 75 %, vcs (100 % + 60 %) / 2 = 80 %
	devices = clock.devices(c)
	if u := devices[0].Utilization; u == nil || *u != 80 {
		t.Errorf("expected 80 %% utilization, got %v", u)
	}
	if devices[0].Temperature != nil || devices[0].MemoryTotal != nil {
		t.Errorf("unexpected statistics: %+v", devices[0])
	}

	// rcs busy longer than the elapsed time is capped
	devices = clock.devices(c)
	if u := devices[0].Utilization; u == nil || *u != 100 {
		t.Errorf("expected 100 %% utilization, got %v", u)
	}

	// Removed GPUs release their counters
	if err := os.Remove(filepath.Join(c.sysRoot, "class", "drm", "card0")); err != nil {
		t.Fatal(err)
	}
	if devices := c.devices(); len(devices) != 0 {
		t.Fatalf("expected no devices, got %+v", devices)
	}
	if perf.openCount() != 0 {
		t.Errorf("expected closed counters, %d are open", perf.openCount())
	}
}

func TestI915ReadError(t *testing.T) {
	c, perf, clock := setupI915(t)
	clock.devices(c)

	perf.readErr = unix.ENODEV
	if devices := clock.devices(c); devices[0].Utilization != nil {
		t.Errorf("expected no utilization: %+v", devices[0])
	}
	if perf.openCount() != 0 {
		t.Errorf("expected closed counters, %d are open", perf.openCount())
	}
}

func TestI915Discrete(t *testing.T) {
	c, perf, clock := setupI915(t)
	addCard(t, c.sysRoot, "card1", "0000:03:00.0", "i915", map[string]string{
		"vendor": "0x8086\n",
		"device": "0x56a0\n",
	})
	addPMU(t, c.sysRoot, "i915_0000_03_00.0", "19", map[string]string{
		"rcs0-busy": "config=0x0",
	})
	perf.values[perfKey{19, 0}] = []uint64{0, 0.5e9}

	clock.devices(c)
	devices := clock.devices(c)
	if len(devices) != 2 {
		t.Fatalf("expected 2 devices, got %+v", devices)
	}
	for _, device := range devices {
		want := 80.0
		if device.Address == "0000:03:00.0" {
			want = 25
		}
		if device.Utilization == nil || *device.Utilization != want {
			t.Errorf("expected %v %% utilization for %s, got %v", want, device.Address, device.Utilization)
		}
	}
}

func TestI915WithoutPermission(t *testing.T) {
	c, _, clock := setupI915(t)
	c.perfOpen = func(uint32, int, uint64) (perfCounter, error) {
		return nil, unix.EACCES
	}
	for range 2 {
		if devices := clock.devices(c); devices[0].Utilization != nil {
			t.Errorf("expected no utilization: %+v", devices[0])
		}
	}
}

func TestI915Concurrent(t *testing.T) {
	c, _, _ := setupI915(t)
	g := gpu{collector: c}

	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			if _, err := g.GetDevices(); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
}

func setupXe(t *testing.T) (*collector, *fakePerf) {
	t.Helper()
	c := newTestCollector(t, "", nil)
	addCard(t, c.sysRoot, "card1", "0000:03:00.0", "xe", map[string]string{
		"vendor":                    "0x8086\n",
		"device":                    "0xe20b\n",
		"power/runtime_status":      "active\n",
		"tile0/gt0/freq0/act_freq":  "2850\n",
		"tile0/gt1/freq0/act_freq":  "0\n",
		"hwmon/hwmon2/power1_input": "40000000\n",
	})
	addPMU(t, c.sysRoot, "xe_0000_03_00.0", "20", map[string]string{
		"engine-active-ticks": "event=0x02",
		"engine-total-ticks":  "event=0x03",
		"gt-c6-residency":     "event=0x01",
	})
	perf := newFakePerf(map[perfKey][]uint64{
		// render on GT 0: 30 %
		{20, xeConfig(2, 0, 0, 0)}: {100, 400},
		{20, xeConfig(3, 0, 0, 0)}: {1000, 2000},
		// copy on GT 0: idle
		{20, xeConfig(2, 0, 1, 0)}: {0, 0},
		{20, xeConfig(3, 0, 1, 0)}: {0, 1000},
		// two video decode engines on the media GT: (90 % + 50 %) / 2
		{20, xeConfig(2, 1, 2, 0)}: {0, 900},
		{20, xeConfig(3, 1, 2, 0)}: {0, 1000},
		{20, xeConfig(2, 1, 2, 1)}: {0, 500},
		{20, xeConfig(3, 1, 2, 1)}: {0, 1000},
		// video enhance without total ticks is left out
		{20, xeConfig(2, 1, 3, 0)}: {0, 1000},
	})
	c.perfOpen = perf.openEvent
	return c, perf
}

func TestXeUtilization(t *testing.T) {
	c, perf := setupXe(t)
	var slept time.Duration
	c.sleep = func(d time.Duration) { slept += d }

	devices := c.devices()
	if len(devices) != 1 {
		t.Fatalf("expected 1 device, got %+v", devices)
	}
	d := devices[0]
	if d.Utilization == nil || *d.Utilization != 70 {
		t.Errorf("expected 70 %% utilization, got %v", d.Utilization)
	}
	if d.Power == nil || *d.Power != 40 {
		t.Errorf("unexpected power %v", d.Power)
	}
	if d.MemoryUsed != nil || d.MemoryTotal != nil {
		t.Errorf("xe has no VRAM statistics in sysfs: %+v", d)
	}
	if slept != xeSampleWindow {
		t.Errorf("expected sample window of %s, slept %s", xeSampleWindow, slept)
	}
	if perf.openCount() != 0 {
		t.Errorf("xe events must be closed after sampling, %d are open", perf.openCount())
	}
}

func TestXeSuspended(t *testing.T) {
	c, _ := setupXe(t)
	writeFile(t, filepath.Join(c.sysRoot, "devices", "pci0000:00", "0000:03:00.0", "power", "runtime_status"), "suspended\n")
	c.perfOpen = func(uint32, int, uint64) (perfCounter, error) {
		t.Error("a suspended GPU must not be woken up")
		return nil, errors.New("unexpected")
	}

	devices := c.devices()
	if u := devices[0].Utilization; u == nil || *u != 0 {
		t.Errorf("expected idle GPU, got %v", u)
	}
}

func TestXeWithoutEngineActivity(t *testing.T) {
	c, _ := setupXe(t)
	// GuC firmware older than interface 1.14.1
	for _, event := range []string{"engine-active-ticks", "engine-total-ticks"} {
		if err := os.Remove(filepath.Join(c.sysRoot, "bus", "event_source", "devices", "xe_0000_03_00.0", "events", event)); err != nil {
			t.Fatal(err)
		}
	}
	if devices := c.devices(); devices[0].Utilization != nil {
		t.Errorf("expected no utilization: %+v", devices[0])
	}
}

func TestXeReadError(t *testing.T) {
	c, perf := setupXe(t)
	perf.readErr = unix.ENODEV
	if devices := c.devices(); devices[0].Utilization != nil {
		t.Errorf("expected no utilization: %+v", devices[0])
	}
	if perf.openCount() != 0 {
		t.Errorf("expected closed events, %d are open", perf.openCount())
	}
}

func TestXeGTs(t *testing.T) {
	root := t.TempDir()
	for _, dir := range []string{"tile0/gt0", "tile0/gt1", "tile1/gt2", "tile0/gtx"} {
		if err := os.MkdirAll(filepath.Join(root, dir), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if gts := xeGTs(root); len(gts) != 3 || gts[0] != 0 || gts[2] != 2 {
		t.Errorf("unexpected GTs %v", gts)
	}
	if gts := xeGTs(t.TempDir()); len(gts) != 1 || gts[0] != 0 {
		t.Errorf("expected GT 0 as fallback, got %v", gts)
	}
}

func TestReadPMU(t *testing.T) {
	c := newTestCollector(t, "", nil)
	addPMU(t, c.sysRoot, "i915", "18", map[string]string{"rcs0-busy": "config=0x0"})
	writeFile(t, filepath.Join(c.sysRoot, "bus", "event_source", "devices", "i915", "cpumask"), "4-7\n")

	p, ok := c.readPMU("i915")
	if !ok || p.pmuType != 18 || p.cpu != 4 {
		t.Errorf("unexpected PMU %+v", p)
	}
	if config, ok := p.eventConfig("rcs0-busy"); !ok || config != 0 {
		t.Errorf("unexpected config %v", config)
	}
	if _, ok := p.eventConfig("missing"); ok {
		t.Error("missing event must not be found")
	}
	if _, ok := c.readPMU("xe_0000_03_00.0"); ok {
		t.Error("missing PMU must not be found")
	}
}
