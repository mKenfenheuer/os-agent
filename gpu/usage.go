package gpu

import (
	"bufio"
	"context"
	"encoding/csv"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/godbus/dbus/v5"

	logging "github.com/home-assistant/os-agent/utils/log"
)

// Sample window of the DRM engine counters per call.
const drmSampleWindow = 200 * time.Millisecond

// Docker container ID in /proc/<pid>/cgroup, with the systemd
// (docker-<id>.scope) or the cgroupfs (/docker/<id>) cgroup driver.
var containerIDRegexp = regexp.MustCompile(`(?:docker-|/docker/)([0-9a-f]{64})`)

// ContainerUsage is the usage of one GPU by the processes of one container.
// Statistics the driver doesn't provide per process are nil.
type ContainerUsage struct {
	Container   string   // full Docker container ID
	Address     string   // PCI address of the GPU
	MemoryUsed  *uint64  // bytes of device memory
	Utilization *float64 // percent of the busiest engine
}

func (u ContainerUsage) toDBus() map[string]dbus.Variant {
	result := map[string]dbus.Variant{
		"Container": dbus.MakeVariant(u.Container),
		"Address":   dbus.MakeVariant(u.Address),
	}
	if u.MemoryUsed != nil {
		result["MemoryUsed"] = dbus.MakeVariant(*u.MemoryUsed)
	}
	if u.Utilization != nil {
		result["Utilization"] = dbus.MakeVariant(*u.Utilization)
	}
	return result
}

type usageKey struct {
	container string
	address   string
}

// containerUsage returns the GPU usage per container and GPU. Processes
// outside of containers are left out.
func (c *collector) containerUsage() []ContainerUsage {
	usage := map[usageKey]*ContainerUsage{}
	var order []usageKey
	get := func(key usageKey) *ContainerUsage {
		if u, ok := usage[key]; ok {
			return u
		}
		u := &ContainerUsage{Container: key.container, Address: key.address}
		usage[key] = u
		order = append(order, key)
		return u
	}

	for _, process := range c.nvidiaProcesses() {
		container := c.containerOf(process.pid)
		if container == "" {
			continue
		}
		u := get(usageKey{container, process.address})
		u.MemoryUsed = addUint(u.MemoryUsed, process.memory)
		if process.utilization != nil {
			// processes of a container share the GPU, so their parts add up
			value := *process.utilization
			if u.Utilization != nil {
				value += *u.Utilization
			}
			value = min(value, 100)
			u.Utilization = &value
		}
	}

	for key, client := range c.drmClients() {
		u := get(key)
		if client.memory != nil {
			u.MemoryUsed = addUint(u.MemoryUsed, *client.memory)
		}
		if client.utilization != nil {
			value := *client.utilization
			if u.Utilization != nil {
				value = max(value, *u.Utilization)
			}
			u.Utilization = &value
		}
	}

	result := make([]ContainerUsage, 0, len(order))
	for _, key := range order {
		result = append(result, *usage[key])
	}
	return result
}

func addUint(sum *uint64, value uint64) *uint64 {
	if sum != nil {
		value += *sum
	}
	return &value
}

// containerOf returns the Docker container ID of a process or "".
func (c *collector) containerOf(pid int) string {
	data, err := os.ReadFile(filepath.Join(c.procRoot, strconv.Itoa(pid), "cgroup"))
	if err != nil {
		return ""
	}
	match := containerIDRegexp.FindSubmatch(data)
	if match == nil {
		return ""
	}
	return string(match[1])
}

type nvidiaProcess struct {
	address     string
	pid         int
	memory      uint64   // bytes
	utilization *float64 // percent of the streaming multiprocessors
}

// nvidiaProcesses lists the compute processes NVML knows with their memory
// and, if the GPU reports it, their utilization.
func (c *collector) nvidiaProcesses() []nvidiaProcess {
	if c.nvidiaSMIApps == nil {
		return nil
	}
	if _, err := os.Stat(filepath.Join(c.procRoot, "driver", "nvidia", "version")); err != nil {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), nvidiaSMITimeout)
	defer cancel()

	output, err := c.nvidiaSMIApps(ctx)
	if err != nil {
		logging.Warning.Printf("Can't query NVIDIA GPU processes: %s", err)
		return nil
	}
	processes, err := parseNvidiaApps(output)
	if err != nil {
		logging.Warning.Printf("Can't parse nvidia-smi output: %s", err)
	}

	utilization := c.nvidiaUtilization(ctx)
	for i := range processes {
		if value, ok := utilization[nvidiaPID{processes[i].address, processes[i].pid}]; ok {
			processes[i].utilization = &value
		}
	}
	return processes
}

type nvidiaPID struct {
	address string
	pid     int
}

// nvidiaUtilization returns the SM utilization per GPU and process of the
// last second. GPUs without these samples (e.g. older GeForce) report "-".
func (c *collector) nvidiaUtilization(ctx context.Context) map[nvidiaPID]float64 {
	if c.nvidiaPmon == nil || c.nvidiaSMIIndex == nil {
		return nil
	}
	indexOutput, err := c.nvidiaSMIIndex(ctx)
	if err != nil {
		return nil
	}
	addresses := map[string]string{}
	reader := csv.NewReader(strings.NewReader(string(indexOutput)))
	reader.TrimLeadingSpace = true
	records, err := reader.ReadAll()
	if err != nil {
		return nil
	}
	for _, record := range records {
		if len(record) == 2 {
			addresses[strings.TrimSpace(record[0])] = normalizePCIAddress(record[1])
		}
	}

	output, err := c.nvidiaPmon(ctx)
	if err != nil {
		return nil
	}
	result := map[nvidiaPID]float64{}
	for _, line := range strings.Split(string(output), "\n") {
		// gpu pid type sm mem enc dec ... command
		fields := strings.Fields(line)
		if len(fields) < 4 || strings.HasPrefix(fields[0], "#") {
			continue
		}
		address, ok := addresses[fields[0]]
		if !ok {
			continue
		}
		pid, err := strconv.Atoi(fields[1])
		if err != nil {
			continue
		}
		if sm := parseFloat(fields[3], 1); sm != nil {
			result[nvidiaPID{address, pid}] = *sm
		}
	}
	return result
}

func parseNvidiaApps(output []byte) ([]nvidiaProcess, error) {
	reader := csv.NewReader(strings.NewReader(string(output)))
	reader.TrimLeadingSpace = true
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	processes := make([]nvidiaProcess, 0, len(records))
	for _, record := range records {
		if len(record) != 3 {
			continue
		}
		pid, err := strconv.Atoi(strings.TrimSpace(record[1]))
		if err != nil {
			continue
		}
		// nvidia-smi reports memory in MiB
		memory := parseFloat(record[2], 1024*1024)
		if memory == nil {
			continue
		}
		processes = append(processes, nvidiaProcess{
			address: normalizePCIAddress(record[0]),
			pid:     pid,
			memory:  *toUint64(*memory),
		})
	}
	return processes, nil
}

// drmFile is an open DRM file of a process, as fdinfo reports it.
type drmFile struct {
	path     string // fdinfo file
	key      usageKey
	clientID string
	memory   *uint64
	engines  map[string]uint64 // busy time in ns
	capacity map[string]uint64 // engines per class
}

type drmUsage struct {
	memory      *uint64
	utilization *float64
}

// drmClients reads the DRM usage statistics (drm-usage-stats) of all
// processes in containers, summed up per container and GPU. A client opened
// through several file descriptors counts once.
func (c *collector) drmClients() map[usageKey]drmUsage {
	files := c.drmFiles()
	if len(files) == 0 {
		return nil
	}

	start := c.now()
	c.sleep(drmSampleWindow)
	elapsed := float64(c.now().Sub(start).Nanoseconds())

	result := map[usageKey]drmUsage{}
	// busy time per container, GPU and engine class
	busy := map[usageKey]map[string]float64{}
	capacity := map[usageKey]map[string]uint64{}
	for _, file := range files {
		usage := result[file.key]
		if file.memory != nil {
			usage.memory = addUint(usage.memory, *file.memory)
		}
		result[file.key] = usage

		later, ok := parseFdinfo(file.path)
		if !ok || later.clientID != file.clientID || elapsed <= 0 {
			continue
		}
		if busy[file.key] == nil {
			busy[file.key] = map[string]float64{}
			capacity[file.key] = map[string]uint64{}
		}
		for engine, value := range later.engines {
			before, ok := file.engines[engine]
			if !ok || value < before {
				continue
			}
			busy[file.key][engine] += float64(value - before)
			capacity[file.key][engine] = max(capacity[file.key][engine], file.capacity[engine])
		}
	}

	for key, engines := range busy {
		var highest *float64
		for engine, value := range engines {
			engineCount := max(capacity[key][engine], 1)
			percent := min(value/elapsed/float64(engineCount)*100, 100)
			if highest == nil || percent > *highest {
				highest = &percent
			}
		}
		usage := result[key]
		usage.utilization = highest
		result[key] = usage
	}
	return result
}

// drmFiles finds the open DRM files of processes in containers. Each DRM
// client is returned once.
func (c *collector) drmFiles() []drmFile {
	processes, err := os.ReadDir(c.procRoot)
	if err != nil {
		return nil
	}

	var files []drmFile
	seen := map[string]bool{}
	for _, process := range processes {
		pid, err := strconv.Atoi(process.Name())
		if err != nil {
			continue
		}
		fdDir := filepath.Join(c.procRoot, process.Name(), "fd")
		fds, err := os.ReadDir(fdDir)
		if err != nil {
			continue
		}
		container := ""
		for _, fd := range fds {
			target, err := os.Readlink(filepath.Join(fdDir, fd.Name()))
			if err != nil || !strings.HasPrefix(target, "/dev/dri/") {
				continue
			}
			if container == "" {
				container = c.containerOf(pid)
				if container == "" {
					break
				}
			}
			path := filepath.Join(c.procRoot, process.Name(), "fdinfo", fd.Name())
			file, ok := parseFdinfo(path)
			if !ok {
				continue
			}
			// The same client can be open through several file descriptors
			// or be shared with child processes
			id := file.key.address + "/" + file.clientID
			if seen[id] {
				continue
			}
			seen[id] = true
			file.key.container = container
			files = append(files, file)
		}
	}
	return files
}

// parseFdinfo reads the DRM keys of an fdinfo file. Files without
// drm-pdev and drm-client-id are no DRM clients of a PCI GPU.
func parseFdinfo(path string) (drmFile, bool) {
	f, err := os.Open(path) //nolint:gosec
	if err != nil {
		return drmFile{}, false
	}
	defer f.Close()

	file := drmFile{
		path:     path,
		engines:  map[string]uint64{},
		capacity: map[string]uint64{},
	}
	var resident, legacy *uint64
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		key, value, found := strings.Cut(scanner.Text(), ":")
		if !found {
			continue
		}
		value = strings.TrimSpace(value)
		switch {
		case key == "drm-pdev":
			file.key.address = value
		case key == "drm-client-id":
			file.clientID = value
		case strings.HasPrefix(key, "drm-engine-capacity-"):
			if n := parseUint(value); n != nil {
				file.capacity[strings.TrimPrefix(key, "drm-engine-capacity-")] = *n
			}
		case strings.HasPrefix(key, "drm-engine-"):
			if n := parseUint(strings.TrimSuffix(value, " ns")); n != nil {
				file.engines[strings.TrimPrefix(key, "drm-engine-")] = *n
			}
		case strings.HasPrefix(key, "drm-resident-"):
			if isDeviceMemory(strings.TrimPrefix(key, "drm-resident-")) {
				resident = addSize(resident, value)
			}
		case strings.HasPrefix(key, "drm-memory-"):
			// Deprecated name of drm-resident-, still used by amdgpu
			if isDeviceMemory(strings.TrimPrefix(key, "drm-memory-")) {
				legacy = addSize(legacy, value)
			}
		}
	}
	if file.key.address == "" || file.clientID == "" {
		return drmFile{}, false
	}
	file.memory = resident
	if file.memory == nil {
		file.memory = legacy
	}
	return file, true
}

// isDeviceMemory tells memory regions on the GPU (amdgpu "vram", xe
// "vram0", i915 "local0") from system memory ("gtt", "system", "cpu").
func isDeviceMemory(region string) bool {
	return strings.HasPrefix(region, "vram") || strings.HasPrefix(region, "local")
}

// addSize adds a size like "1024 KiB" to sum.
func addSize(sum *uint64, value string) *uint64 {
	number, unit, _ := strings.Cut(value, " ")
	n := parseUint(number)
	if n == nil {
		return sum
	}
	switch unit {
	case "KiB":
		*n *= 1024
	case "MiB":
		*n *= 1024 * 1024
	case "GiB":
		*n *= 1024 * 1024 * 1024
	}
	return addUint(sum, *n)
}
