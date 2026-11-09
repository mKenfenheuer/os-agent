// Package gpu exposes information and live statistics of the GPUs of the
// host, for vendors whose user space can't run in the Supervisor or Home
// Assistant containers (e.g. NVML of the NVIDIA driver needs glibc).
package gpu

import (
	"github.com/godbus/dbus/v5"
	"github.com/godbus/dbus/v5/introspect"

	logging "github.com/home-assistant/os-agent/utils/log"
)

const (
	objectPath = "/io/hass/os/GPU"
	ifaceName  = "io.hass.os.GPU"
)

// Device describes a GPU. Statistics the driver doesn't provide are nil.
type Device struct {
	Address       string
	Vendor        string
	Name          string
	Driver        string
	DriverVersion string
	Utilization   *float64 // percent
	MemoryUsed    *uint64  // bytes
	MemoryTotal   *uint64  // bytes
	Temperature   *float64 // degree Celsius
	Power         *float64 // watts
}

// toDBus converts the device into a dictionary, leaving out unknown values.
func (d Device) toDBus() map[string]dbus.Variant {
	result := map[string]dbus.Variant{
		"Address":       dbus.MakeVariant(d.Address),
		"Vendor":        dbus.MakeVariant(d.Vendor),
		"Name":          dbus.MakeVariant(d.Name),
		"Driver":        dbus.MakeVariant(d.Driver),
		"DriverVersion": dbus.MakeVariant(d.DriverVersion),
	}
	if d.Utilization != nil {
		result["Utilization"] = dbus.MakeVariant(*d.Utilization)
	}
	if d.MemoryUsed != nil {
		result["MemoryUsed"] = dbus.MakeVariant(*d.MemoryUsed)
	}
	if d.MemoryTotal != nil {
		result["MemoryTotal"] = dbus.MakeVariant(*d.MemoryTotal)
	}
	if d.Temperature != nil {
		result["Temperature"] = dbus.MakeVariant(*d.Temperature)
	}
	if d.Power != nil {
		result["Power"] = dbus.MakeVariant(*d.Power)
	}
	return result
}

type gpu struct {
	collector collector
}

// GetDevices returns all GPUs with their current statistics.
func (g gpu) GetDevices() ([]map[string]dbus.Variant, *dbus.Error) {
	devices := g.collector.devices()

	result := make([]map[string]dbus.Variant, 0, len(devices))
	for _, device := range devices {
		result = append(result, device.toDBus())
	}
	return result, nil
}

func InitializeDBus(conn *dbus.Conn) {
	g := gpu{collector: newCollector()}

	err := conn.Export(g, objectPath, ifaceName)
	if err != nil {
		logging.Critical.Panic(err)
	}

	node := &introspect.Node{
		Name: objectPath,
		Interfaces: []introspect.Interface{
			introspect.IntrospectData,
			{
				Name:    ifaceName,
				Methods: introspect.Methods(g),
			},
		},
	}

	err = conn.Export(introspect.NewIntrospectable(node), objectPath, "org.freedesktop.DBus.Introspectable")
	if err != nil {
		logging.Critical.Panic(err)
	}

	logging.Info.Printf("Exposing object %s with interface %s ...", objectPath, ifaceName)
}
