package gpu

import (
	"context"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

const (
	containerA = "1111111111111111111111111111111111111111111111111111111111111111"
	containerB = "2222222222222222222222222222222222222222222222222222222222222222"
)

// addProcess creates /proc/<pid> with a cgroup file and open DRM files.
// fdinfo maps the file descriptor number to the content of its fdinfo file.
func addProcess(t *testing.T, procRoot string, pid int, cgroup string, fdinfo map[int]string) {
	t.Helper()
	dir := filepath.Join(procRoot, strconv.Itoa(pid))
	writeFile(t, filepath.Join(dir, "cgroup"), cgroup)
	symlink(t, "/dev/null", filepath.Join(dir, "fd", "0"))
	writeFile(t, filepath.Join(dir, "fdinfo", "0"), "pos:\t0\n")
	for fd, content := range fdinfo {
		symlink(t, "/dev/dri/renderD128", filepath.Join(dir, "fd", strconv.Itoa(fd)))
		writeFile(t, filepath.Join(dir, "fdinfo", strconv.Itoa(fd)), content)
	}
}

func dockerCgroup(container string) string {
	return "0::/system.slice/docker-" + container + ".scope\n"
}

func amdgpuFdinfo(clientID string, vramKiB, gfxNs int) string {
	return "pos:\t0\nflags:\t02100002\ndrm-driver:\tamdgpu\n" +
		"drm-pdev:\t0000:03:00.0\ndrm-client-id:\t" + clientID + "\n" +
		"drm-memory-vram:\t" + strconv.Itoa(vramKiB) + " KiB\n" +
		"drm-memory-gtt:\t2048 KiB\n" +
		"drm-engine-gfx:\t" + strconv.Itoa(gfxNs) + " ns\n" +
		"drm-engine-compute:\t0 ns\n"
}

func findUsage(usage []ContainerUsage, container, address string) *ContainerUsage {
	for i := range usage {
		if usage[i].Container == container && usage[i].Address == address {
			return &usage[i]
		}
	}
	return nil
}

func TestContainerUsageDRM(t *testing.T) {
	c := newTestCollector(t, "", nil)
	// Container A: two processes, client 10 open twice, client 11 once
	addProcess(t, c.procRoot, 100, dockerCgroup(containerA), map[int]string{
		5: amdgpuFdinfo("10", 1024*1024, 1000),
		6: amdgpuFdinfo("10", 1024*1024, 1000),
	})
	addProcess(t, c.procRoot, 101, "0::/docker/"+containerA+"\n", map[int]string{
		5: amdgpuFdinfo("11", 512*1024, 0),
	})
	// Process on the host, not in a container
	addProcess(t, c.procRoot, 200, "0::/system.slice/os-agent.service\n", map[int]string{
		5: amdgpuFdinfo("12", 4*1024*1024, 0),
	})

	// The engine counter of client 10 advances by half the sample window
	start := time.Unix(1000, 0)
	now := start
	c.now = func() time.Time { return now }
	c.sleep = func(d time.Duration) {
		now = now.Add(d)
		writeFile(t, filepath.Join(c.procRoot, "100", "fdinfo", "5"),
			amdgpuFdinfo("10", 1024*1024, 1000+int(d.Nanoseconds()/2)))
	}

	usage := c.containerUsage()
	if len(usage) != 1 {
		t.Fatalf("expected usage of one container, got %+v", usage)
	}
	u := findUsage(usage, containerA, "0000:03:00.0")
	if u == nil {
		t.Fatalf("no usage of container A: %+v", usage)
	}
	if u.MemoryUsed == nil || *u.MemoryUsed != 1536*1024*1024 {
		t.Errorf("memory: %v", u.MemoryUsed)
	}
	if u.Utilization == nil || *u.Utilization != 50 {
		t.Errorf("utilization: %v", u.Utilization)
	}
}

func TestContainerUsageResidentAndCapacity(t *testing.T) {
	c := newTestCollector(t, "", nil)
	fdinfo := func(rcsNs int) string {
		return "drm-driver:\txe\ndrm-pdev:\t0000:00:02.0\ndrm-client-id:\t7\n" +
			"drm-resident-vram0:\t100 MiB\ndrm-memory-vram0:\t999 MiB\n" +
			"drm-resident-system:\t50 MiB\n" +
			"drm-engine-capacity-vcs:\t2\n" +
			"drm-engine-vcs:\t" + strconv.Itoa(rcsNs) + " ns\n"
	}
	addProcess(t, c.procRoot, 300, dockerCgroup(containerB), map[int]string{3: fdinfo(0)})

	now := time.Unix(1000, 0)
	c.now = func() time.Time { return now }
	c.sleep = func(d time.Duration) {
		now = now.Add(d)
		// one of two video engines busy all the time
		writeFile(t, filepath.Join(c.procRoot, "300", "fdinfo", "3"), fdinfo(int(d.Nanoseconds())))
	}

	u := findUsage(c.containerUsage(), containerB, "0000:00:02.0")
	if u == nil {
		t.Fatal("no usage of container B")
	}
	// drm-resident- wins over the deprecated drm-memory-, system memory is left out
	if u.MemoryUsed == nil || *u.MemoryUsed != 100*1024*1024 {
		t.Errorf("memory: %v", u.MemoryUsed)
	}
	if u.Utilization == nil || *u.Utilization != 50 {
		t.Errorf("utilization: %v", u.Utilization)
	}
}

func TestContainerUsageNVIDIA(t *testing.T) {
	c := newTestCollector(t, "", nil)
	writeFile(t, filepath.Join(c.procRoot, "driver", "nvidia", "version"), "NVRM version: 580.196.07\n")
	c.nvidiaSMIApps = func(context.Context) ([]byte, error) {
		return []byte("00000000:01:00.0, 12025, 2294\n" +
			"00000000:01:00.0, 12326, 910\n" +
			"00000000:01:00.0, 999, 100\n" +
			"00000000:01:00.0, 12486, [N/A]\n"), nil
	}
	c.nvidiaSMIIndex = func(context.Context) ([]byte, error) {
		return []byte("0, 00000000:01:00.0, 42\n"), nil
	}
	c.nvidiaPmon = func(context.Context) ([]byte, error) {
		return []byte("# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command \n" +
			"# Idx           #    C/G      %      %      %      %      %      %    name \n" +
			"    0      12025     C     60     30      -      -      -      -    llama-server\n" +
			"    0      12326     C     15      5      -      -      -      -    whisper-server\n" +
			"    0        999     C     20      1      -      -      -      -    python\n"), nil
	}
	writeFile(t, filepath.Join(c.procRoot, "12025", "cgroup"), dockerCgroup(containerA))
	writeFile(t, filepath.Join(c.procRoot, "12326", "cgroup"), dockerCgroup(containerA))
	writeFile(t, filepath.Join(c.procRoot, "999", "cgroup"), "0::/init.scope\n")

	usage := c.containerUsage()
	if len(usage) != 1 {
		t.Fatalf("expected usage of one container, got %+v", usage)
	}
	u := findUsage(usage, containerA, "0000:01:00.0")
	if u == nil {
		t.Fatalf("no usage of container A: %+v", usage)
	}
	if u.MemoryUsed == nil || *u.MemoryUsed != (2294+910)*1024*1024 {
		t.Errorf("memory: %v", u.MemoryUsed)
	}
	if u.Utilization == nil || *u.Utilization != 75 {
		t.Errorf("utilization: %v", u.Utilization)
	}
}

func TestContainerUsageNVIDIAWithoutSamples(t *testing.T) {
	c := newTestCollector(t, "", nil)
	writeFile(t, filepath.Join(c.procRoot, "driver", "nvidia", "version"), "NVRM version: 580.196.07\n")
	c.nvidiaSMIApps = func(context.Context) ([]byte, error) {
		return []byte("00000000:01:00.0, 12025, 2294\n"), nil
	}
	c.nvidiaSMIIndex = func(context.Context) ([]byte, error) {
		return []byte("0, 00000000:01:00.0, 42\n"), nil
	}
	// GPUs without per-process samples report "-", although the GPU is busy
	c.nvidiaPmon = func(context.Context) ([]byte, error) {
		return []byte("    0      12025     C      -      -      -      -      -      -    llama-server\n"), nil
	}
	writeFile(t, filepath.Join(c.procRoot, "12025", "cgroup"), dockerCgroup(containerA))

	u := findUsage(c.containerUsage(), containerA, "0000:01:00.0")
	if u == nil || u.MemoryUsed == nil {
		t.Fatalf("no usage of container A: %+v", u)
	}
	if u.Utilization != nil {
		t.Errorf("utilization without samples: %v", *u.Utilization)
	}
	if _, ok := u.toDBus()["Utilization"]; ok {
		t.Error("unknown utilization must be left out on D-Bus")
	}
}

func TestContainerUsageNVIDIAIdle(t *testing.T) {
	for name, tc := range map[string]struct {
		gpuUtilization string
		pmon           string
	}{
		// another process has a sample, so "-" means no work in the last second
		"other process sampled": {"5", "    0       5884     C      -      -      -      -      -      -    whisper-server \n" +
			"    0      18664     C      5      0      -      -      -      -    llama-server   \n"},
		// no process has a sample, but the whole GPU is idle
		"idle GPU": {"0", "    0       5884     C      -      -      -      -      -      -    whisper-server \n"},
	} {
		t.Run(name, func(t *testing.T) {
			c := newTestCollector(t, "", nil)
			writeFile(t, filepath.Join(c.procRoot, "driver", "nvidia", "version"), "NVRM version: 580.196.07\n")
			c.nvidiaSMIApps = func(context.Context) ([]byte, error) {
				return []byte("00000000:01:00.0, 5884, 910\n"), nil
			}
			c.nvidiaSMIIndex = func(context.Context) ([]byte, error) {
				return []byte("0, 00000000:01:00.0, " + tc.gpuUtilization + "\n"), nil
			}
			c.nvidiaPmon = func(context.Context) ([]byte, error) { return []byte(tc.pmon), nil }
			writeFile(t, filepath.Join(c.procRoot, "5884", "cgroup"), dockerCgroup(containerA))

			u := findUsage(c.containerUsage(), containerA, "0000:01:00.0")
			if u == nil || u.Utilization == nil || *u.Utilization != 0 {
				t.Fatalf("expected idle container, got %+v", u)
			}
		})
	}
}

func TestContainerUsageWithoutGPU(t *testing.T) {
	c := newTestCollector(t, "", nil)
	addProcess(t, c.procRoot, 100, dockerCgroup(containerA), nil)
	if usage := c.containerUsage(); len(usage) != 0 {
		t.Errorf("expected no usage, got %+v", usage)
	}
}
