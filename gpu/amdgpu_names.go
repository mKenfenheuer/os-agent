package gpu

import (
	_ "embed"
	"strings"
	"sync"
)

// amdgpu.ids maps PCI device and revision IDs to marketing names, amdgpu
// doesn't expose them in sysfs. Taken from libdrm (MIT license):
// https://gitlab.freedesktop.org/mesa/drm/-/blob/main/data/amdgpu.ids
//
//go:embed amdgpu.ids
var amdgpuIDs string

type amdgpuID struct {
	device, revision string
}

type amdgpuSubsystem struct {
	amdgpuID
	vendor, device string
}

var (
	amdgpuNamesOnce      sync.Once
	amdgpuNames          map[amdgpuID]string
	amdgpuSubsystemNames map[amdgpuSubsystem]string
)

func parseAMDGPUIDs(data string) (map[amdgpuID]string, map[amdgpuSubsystem]string) {
	names := map[amdgpuID]string{}
	subsystemNames := map[amdgpuSubsystem]string{}
	var last *amdgpuID

	for _, line := range strings.Split(data, "\n") {
		if strings.HasPrefix(line, "#") || strings.TrimSpace(line) == "" {
			continue
		}
		child := strings.HasPrefix(line, "\t")
		fields := strings.SplitN(strings.TrimSpace(line), ",", 3)
		if len(fields) != 3 {
			// File version line
			continue
		}
		first := strings.ToUpper(strings.TrimSpace(fields[0]))
		second := strings.ToUpper(strings.TrimSpace(fields[1]))
		name := strings.TrimSpace(fields[2])

		if child {
			if last != nil {
				subsystemNames[amdgpuSubsystem{*last, first, second}] = name
			}
			continue
		}
		id := amdgpuID{first, second}
		names[id] = name
		last = &id
	}
	return names, subsystemNames
}

// amdgpuName returns the marketing name of an AMD GPU, IDs as in sysfs
// (e.g. "0x73bf", "0xc0"), or "" if unknown.
func amdgpuName(device, revision, subsystemVendor, subsystemDevice string) string {
	amdgpuNamesOnce.Do(func() {
		amdgpuNames, amdgpuSubsystemNames = parseAMDGPUIDs(amdgpuIDs)
	})

	hex := func(value string) string {
		return strings.ToUpper(strings.TrimPrefix(strings.TrimSpace(value), "0x"))
	}
	id := amdgpuID{hex(device), hex(revision)}
	if name, ok := amdgpuSubsystemNames[amdgpuSubsystem{id, hex(subsystemVendor), hex(subsystemDevice)}]; ok {
		return name
	}
	return amdgpuNames[id]
}
