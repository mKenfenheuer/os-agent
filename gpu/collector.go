package gpu

import (
	"context"
	"encoding/csv"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	logging "github.com/home-assistant/os-agent/utils/log"
)

const (
	vendorAMD    = "amd"
	vendorIntel  = "intel"
	vendorNVIDIA = "nvidia"

	nvidiaSMITimeout = 5 * time.Second
)

var pciVendors = map[string]string{
	"0x1002": vendorAMD,
	"0x8086": vendorIntel,
	"0x10de": vendorNVIDIA,
}

var vendorNames = map[string]string{
	vendorAMD:    "AMD",
	vendorIntel:  "Intel",
	vendorNVIDIA: "NVIDIA",
}

// Display-only or emulated adapters, which are no GPUs for compute.
var ignoredDrivers = []string{
	"ast",
	"bochs-drm",
	"cirrus-qemu",
	"efi-framebuffer",
	"hyperv_drm",
	"mgag200",
	"qxl",
	"simple-framebuffer",
	"simpledrm",
	"vboxvideo",
	"vc4-drm",
	"virtio_gpu",
	"vmwgfx",
}

var cardRegexp = regexp.MustCompile(`^card[0-9]+$`)

type collector struct {
	sysRoot   string
	procRoot  string
	nvidiaSMI func(ctx context.Context) ([]byte, error)
	// Compute processes: GPU address, PID and memory in MiB
	nvidiaSMIApps func(ctx context.Context) ([]byte, error)
	// Utilization per process of the last second, by GPU index
	nvidiaPmon func(ctx context.Context) ([]byte, error)
	// GPU index and address, to match the output of nvidiaPmon
	nvidiaSMIIndex func(ctx context.Context) ([]byte, error)
	perfOpen  perfOpenFunc
	now       func() time.Time
	sleep     func(time.Duration)

	// Open busy counters of i915 devices by PCI address
	mu   sync.Mutex
	i915 map[string]*i915Counters
}

func newCollector() *collector {
	return &collector{
		sysRoot:  "/sys",
		procRoot: "/proc",
		perfOpen: openPerfEvent,
		now:      time.Now,
		sleep:    time.Sleep,
		nvidiaSMI: func(ctx context.Context) ([]byte, error) {
			path, err := exec.LookPath("nvidia-smi")
			if err != nil {
				return nil, nil //nolint:nilerr // no NVIDIA driver installed
			}
			return exec.CommandContext(ctx, path, //nolint:gosec
				"--query-gpu=pci.bus_id,name,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
				"--format=csv,noheader,nounits").Output()
		},
		nvidiaSMIApps: func(ctx context.Context) ([]byte, error) {
			return runNvidiaSMI(ctx, "--query-compute-apps=gpu_bus_id,pid,used_memory",
				"--format=csv,noheader,nounits")
		},
		nvidiaPmon: func(ctx context.Context) ([]byte, error) {
			return runNvidiaSMI(ctx, "pmon", "--count", "1", "--select", "u")
		},
		nvidiaSMIIndex: func(ctx context.Context) ([]byte, error) {
			return runNvidiaSMI(ctx, "--query-gpu=index,pci.bus_id", "--format=csv,noheader")
		},
	}
}

func runNvidiaSMI(ctx context.Context, args ...string) ([]byte, error) {
	path, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return nil, nil //nolint:nilerr // no NVIDIA driver installed
	}
	return exec.CommandContext(ctx, path, args...).Output() //nolint:gosec
}

func (c *collector) devices() []Device {
	devices := c.nvidiaDevices()
	devices = append(devices, c.drmDevices()...)
	return devices
}

// nvidiaDevices queries NVML through nvidia-smi.
func (c *collector) nvidiaDevices() []Device {
	if _, err := os.Stat(filepath.Join(c.procRoot, "driver", "nvidia", "version")); err != nil {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), nvidiaSMITimeout)
	defer cancel()

	output, err := c.nvidiaSMI(ctx)
	if err != nil {
		logging.Warning.Printf("Can't query NVIDIA GPUs: %s", err)
		return nil
	}
	devices, err := parseNvidiaSMI(output)
	if err != nil {
		logging.Warning.Printf("Can't parse nvidia-smi output: %s", err)
	}
	return devices
}

func parseNvidiaSMI(output []byte) ([]Device, error) {
	reader := csv.NewReader(strings.NewReader(string(output)))
	reader.TrimLeadingSpace = true
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	devices := make([]Device, 0, len(records))
	for _, record := range records {
		if len(record) != 8 {
			return devices, fmt.Errorf("unexpected number of fields: %d", len(record))
		}

		device := Device{
			Address:       normalizePCIAddress(record[0]),
			Vendor:        vendorNVIDIA,
			Name:          record[1],
			Driver:        "nvidia",
			DriverVersion: record[2],
			Utilization:   parseFloat(record[3], 1),
			Temperature:   parseFloat(record[6], 1),
			Power:         parseFloat(record[7], 1),
		}
		// nvidia-smi reports memory in MiB
		if used := parseFloat(record[4], 1024*1024); used != nil {
			device.MemoryUsed = toUint64(*used)
		}
		if total := parseFloat(record[5], 1024*1024); total != nil {
			device.MemoryTotal = toUint64(*total)
		}
		devices = append(devices, device)
	}
	return devices, nil
}

// normalizePCIAddress converts the 8 digit PCI domain used by NVML
// (00000000:01:00.0) into the format of the kernel (0000:01:00.0).
func normalizePCIAddress(address string) string {
	address = strings.ToLower(strings.TrimSpace(address))
	domain, rest, found := strings.Cut(address, ":")
	if found && len(domain) > 4 {
		domain = domain[len(domain)-4:]
		return domain + ":" + rest
	}
	return address
}

// drmDevices reads GPUs from DRM in sysfs. NVIDIA GPUs are left out, they
// are covered by NVML.
func (c *collector) drmDevices() []Device {
	cards, err := os.ReadDir(filepath.Join(c.sysRoot, "class", "drm"))
	if err != nil {
		return nil
	}

	var devices []Device
	var i915Devices []string
	for _, card := range cards {
		if !cardRegexp.MatchString(card.Name()) {
			continue
		}
		devicePath := filepath.Join(c.sysRoot, "class", "drm", card.Name(), "device")
		device, ok := c.drmDevice(devicePath)
		if ok {
			devices = append(devices, device)
			if device.Driver == "i915" {
				i915Devices = append(i915Devices, device.Address)
			}
		}
	}
	c.pruneI915(i915Devices)
	return devices
}

func (c *collector) drmDevice(devicePath string) (Device, bool) {
	driverPath, err := filepath.EvalSymlinks(filepath.Join(devicePath, "driver"))
	if err != nil {
		return Device{}, false
	}
	driver := filepath.Base(driverPath)
	if slices.Contains(ignoredDrivers, driver) {
		return Device{}, false
	}

	resolved, err := filepath.EvalSymlinks(devicePath)
	if err != nil {
		return Device{}, false
	}

	vendorID := readString(filepath.Join(devicePath, "vendor"))
	vendor := pciVendors[vendorID]
	if vendor == vendorNVIDIA {
		return Device{}, false
	}

	device := Device{
		Address:       filepath.Base(resolved),
		Vendor:        vendor,
		Driver:        driver,
		DriverVersion: c.driverVersion(driver),
		Name:          c.deviceName(devicePath, vendor, vendorID),
	}

	if vendor == vendorAMD {
		if busy := parseFloat(readString(filepath.Join(devicePath, "gpu_busy_percent")), 1); busy != nil {
			device.Utilization = busy
		}
		device.MemoryUsed = parseUint(readString(filepath.Join(devicePath, "mem_info_vram_used")))
		device.MemoryTotal = parseUint(readString(filepath.Join(devicePath, "mem_info_vram_total")))
	}

	if vendor == vendorIntel {
		switch driver {
		case "i915":
			device.Utilization = c.i915Utilization(device.Address)
		case "xe":
			device.Utilization = c.xeUtilization(devicePath, device.Address)
		}
	}

	if hwmon := firstHwmon(devicePath); hwmon != "" {
		// millidegree Celsius
		device.Temperature = parseFloat(readString(filepath.Join(hwmon, "temp1_input")), 0.001)
		// microwatts, average on older and current on newer GPUs
		for _, name := range []string{"power1_average", "power1_input"} {
			if power := parseFloat(readString(filepath.Join(hwmon, name)), 0.000001); power != nil {
				device.Power = power
				break
			}
		}
	}

	return device, true
}

func (c *collector) driverVersion(driver string) string {
	// Out-of-tree drivers report their own version
	if version := readString(filepath.Join(c.sysRoot, "module", strings.ReplaceAll(driver, "-", "_"), "version")); version != "" {
		return version
	}
	return readString(filepath.Join(c.procRoot, "sys", "kernel", "osrelease"))
}

func (c *collector) deviceName(devicePath, vendor, vendorID string) string {
	if name := readString(filepath.Join(devicePath, "product_name")); name != "" {
		return name
	}
	if vendor == vendorAMD {
		if name := amdgpuName(
			readString(filepath.Join(devicePath, "device")),
			readString(filepath.Join(devicePath, "revision")),
			readString(filepath.Join(devicePath, "subsystem_vendor")),
			readString(filepath.Join(devicePath, "subsystem_device")),
		); name != "" {
			return name
		}
	}
	if vendorID != "" {
		deviceID := strings.TrimPrefix(readString(filepath.Join(devicePath, "device")), "0x")
		prefix := vendorNames[vendor]
		if prefix == "" {
			prefix = "PCI"
		}
		return fmt.Sprintf("%s GPU [%s:%s]", prefix, strings.TrimPrefix(vendorID, "0x"), deviceID)
	}
	// Platform devices, e.g. Raspberry Pi V3D: use the device tree compatible
	compatible := readString(filepath.Join(devicePath, "of_node", "compatible"))
	if compatible != "" {
		return strings.SplitN(compatible, "\x00", 2)[0]
	}
	return filepath.Base(devicePath)
}

func firstHwmon(devicePath string) string {
	matches, _ := filepath.Glob(filepath.Join(devicePath, "hwmon", "hwmon*"))
	if len(matches) == 0 {
		return ""
	}
	slices.Sort(matches)
	return matches[0]
}

func readString(path string) string {
	data, err := os.ReadFile(path) //nolint:gosec
	if err != nil {
		return ""
	}
	return strings.TrimRight(strings.TrimSpace(string(data)), "\x00")
}

// parseFloat parses a number and multiplies it by scale. Values NVML marks
// as unavailable ("[N/A]", "[Not Supported]") return nil.
func parseFloat(value string, scale float64) *float64 {
	result, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
	if err != nil {
		return nil
	}
	result *= scale
	return &result
}

func parseUint(value string) *uint64 {
	result, err := strconv.ParseUint(strings.TrimSpace(value), 10, 64)
	if err != nil {
		return nil
	}
	return &result
}

func toUint64(value float64) *uint64 {
	if value < 0 {
		return nil
	}
	result := uint64(value)
	return &result
}
