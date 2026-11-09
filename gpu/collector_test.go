package gpu

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/godbus/dbus/v5"
)

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil { //nolint:gosec
		t.Fatal(err)
	}
}

func symlink(t *testing.T, target, link string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
}

// addCard creates /sys/class/drm/<card> pointing to a device bound to driver.
func addCard(t *testing.T, sysRoot, card, deviceName, driver string, files map[string]string) string {
	t.Helper()
	devicePath := filepath.Join(sysRoot, "devices", "pci0000:00", deviceName)
	symlink(t, filepath.Join(sysRoot, "bus", "pci", "drivers", driver), filepath.Join(devicePath, "driver"))
	if err := os.MkdirAll(filepath.Join(sysRoot, "bus", "pci", "drivers", driver), 0o755); err != nil {
		t.Fatal(err)
	}
	symlink(t, devicePath, filepath.Join(sysRoot, "devices", "drm", card, "device"))
	symlink(t, filepath.Join(sysRoot, "devices", "drm", card), filepath.Join(sysRoot, "class", "drm", card))
	for name, content := range files {
		writeFile(t, filepath.Join(devicePath, name), content)
	}
	return devicePath
}

func newTestCollector(t *testing.T, nvidiaOutput string, nvidiaErr error) *collector {
	t.Helper()
	root := t.TempDir()
	c := &collector{
		sysRoot:  filepath.Join(root, "sys"),
		procRoot: filepath.Join(root, "proc"),
		nvidiaSMI: func(context.Context) ([]byte, error) {
			return []byte(nvidiaOutput), nvidiaErr
		},
		perfOpen: func(uint32, int, uint64) (perfCounter, error) {
			return nil, errors.New("no perf in tests")
		},
		now:   time.Now,
		sleep: func(time.Duration) {},
	}
	writeFile(t, filepath.Join(c.procRoot, "sys", "kernel", "osrelease"), "6.18.60-haos\n")
	return c
}

func TestAMDDevice(t *testing.T) {
	c := newTestCollector(t, "", nil)
	addCard(t, c.sysRoot, "card1", "0000:03:00.0", "amdgpu", map[string]string{
		"vendor":                    "0x1002\n",
		"device":                    "0x744c\n",
		"gpu_busy_percent":          "42\n",
		"mem_info_vram_used":        "1073741824\n",
		"mem_info_vram_total":       "25753026560\n",
		"hwmon/hwmon3/temp1_input":  "51000\n",
		"hwmon/hwmon3/power1_input": "35000000\n",
		"hwmon/hwmon3/temp2_input":  "60000\n",
		"product_name":              "\n",
	})
	// Connectors are no cards
	writeFile(t, filepath.Join(c.sysRoot, "class", "drm", "card1-DP-1", "status"), "connected\n")
	addCard(t, c.sysRoot, "card0", "0000:00:02.0", "simpledrm", nil)

	devices := c.devices()
	if len(devices) != 1 {
		t.Fatalf("expected 1 device, got %d: %+v", len(devices), devices)
	}
	d := devices[0]
	if d.Address != "0000:03:00.0" || d.Vendor != vendorAMD || d.Driver != "amdgpu" {
		t.Errorf("unexpected device: %+v", d)
	}
	if d.Name != "AMD GPU [1002:744c]" {
		t.Errorf("unexpected name %q", d.Name)
	}
	if d.DriverVersion != "6.18.60-haos" {
		t.Errorf("unexpected driver version %q", d.DriverVersion)
	}
	if *d.Utilization != 42 || *d.MemoryUsed != 1073741824 || *d.MemoryTotal != 25753026560 {
		t.Errorf("unexpected statistics: %+v", d)
	}
	if *d.Temperature != 51 || *d.Power != 35 {
		t.Errorf("unexpected temperature %v or power %v", *d.Temperature, *d.Power)
	}
}

func TestIntelDeviceWithoutStatistics(t *testing.T) {
	c := newTestCollector(t, "", nil)
	addCard(t, c.sysRoot, "card0", "0000:00:02.0", "i915", map[string]string{
		"vendor": "0x8086\n",
		"device": "0x46a6\n",
	})

	devices := c.devices()
	if len(devices) != 1 {
		t.Fatalf("expected 1 device, got %d", len(devices))
	}
	d := devices[0]
	if d.Name != "Intel GPU [8086:46a6]" || d.Vendor != vendorIntel {
		t.Errorf("unexpected device: %+v", d)
	}
	if d.Utilization != nil || d.MemoryUsed != nil || d.Temperature != nil || d.Power != nil {
		t.Errorf("expected no statistics: %+v", d)
	}
	if _, ok := d.toDBus()["Utilization"]; ok {
		t.Error("unknown values must be left out on D-Bus")
	}
}

func TestPlatformDevice(t *testing.T) {
	c := newTestCollector(t, "", nil)
	addCard(t, c.sysRoot, "card0", "1002000000.v3d", "v3d", map[string]string{
		"of_node/compatible": "brcm,2712-v3d\x00brcm,2711-v3d\x00",
	})
	addCard(t, c.sysRoot, "card1", "axi:gpu", "vc4-drm", nil)

	devices := c.devices()
	if len(devices) != 1 || devices[0].Name != "brcm,2712-v3d" || devices[0].Vendor != "" {
		t.Fatalf("unexpected devices: %+v", devices)
	}
}

func TestNvidiaDevices(t *testing.T) {
	output := "00000000:01:00.0, NVIDIA GeForce RTX 3090, 615.78.08, 7, 2844, 24576, 34, 16.33\n" +
		"00000000:02:00.0, Tesla T4, 615.78.08, [N/A], 0, 15360, 40, [N/A]\n"
	c := newTestCollector(t, output, nil)
	writeFile(t, filepath.Join(c.procRoot, "driver", "nvidia", "version"), "NVRM version: 615.78.08\n")
	// The DRM card of an NVIDIA GPU must not show up twice
	addCard(t, c.sysRoot, "card1", "0000:01:00.0", "nvidia", map[string]string{"vendor": "0x10de\n"})

	devices := c.devices()
	if len(devices) != 2 {
		t.Fatalf("expected 2 devices, got %d: %+v", len(devices), devices)
	}
	d := devices[0]
	if d.Address != "0000:01:00.0" || d.Name != "NVIDIA GeForce RTX 3090" || d.DriverVersion != "615.78.08" {
		t.Errorf("unexpected device: %+v", d)
	}
	if *d.Utilization != 7 || *d.MemoryUsed != 2844*1024*1024 || *d.MemoryTotal != 24576*1024*1024 {
		t.Errorf("unexpected statistics: %+v", d)
	}
	if *d.Temperature != 34 || *d.Power != 16.33 {
		t.Errorf("unexpected temperature or power: %+v", d)
	}
	if devices[1].Utilization != nil || devices[1].Power != nil {
		t.Errorf("unavailable values must be nil: %+v", devices[1])
	}
}

func TestNvidiaNotLoaded(t *testing.T) {
	c := newTestCollector(t, "", errors.New("must not be called"))
	if devices := c.devices(); len(devices) != 0 {
		t.Fatalf("expected no devices, got %+v", devices)
	}
}

func TestNvidiaSMIError(t *testing.T) {
	c := newTestCollector(t, "", errors.New("exit status 9"))
	writeFile(t, filepath.Join(c.procRoot, "driver", "nvidia", "version"), "NVRM version: 615.78.08\n")
	if devices := c.devices(); len(devices) != 0 {
		t.Fatalf("expected no devices, got %+v", devices)
	}
}

func TestGetDevices(t *testing.T) {
	c := newTestCollector(t, "00000000:01:00.0, NVIDIA RTX A2000, 615.78.08, 1, 2, 3, 4, 5\n", nil)
	writeFile(t, filepath.Join(c.procRoot, "driver", "nvidia", "version"), "NVRM version: 615.78.08\n")

	result, dbusErr := gpu{collector: c}.GetDevices()
	if dbusErr != nil {
		t.Fatal(dbusErr)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 device, got %d", len(result))
	}
	if result[0]["Name"] != dbus.MakeVariant("NVIDIA RTX A2000") {
		t.Errorf("unexpected name %v", result[0]["Name"])
	}
	if result[0]["MemoryTotal"] != dbus.MakeVariant(uint64(3*1024*1024)) {
		t.Errorf("unexpected memory total %v", result[0]["MemoryTotal"])
	}
}

func TestNormalizePCIAddress(t *testing.T) {
	for input, want := range map[string]string{
		"00000000:01:00.0": "0000:01:00.0",
		"0000:0A:00.0":     "0000:0a:00.0",
		"invalid":          "invalid",
	} {
		if got := normalizePCIAddress(input); got != want {
			t.Errorf("normalizePCIAddress(%q) = %q, want %q", input, got, want)
		}
	}
}
