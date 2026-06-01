// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from "vitest";

vi.mock("../adapters/docker", () => ({
  dockerInfoFormat: vi.fn(),
}));

import {
  createArm64WslDockerDesktopGpuProver,
  detectWslDockerDesktopStatus,
  isWslDockerDesktopRuntime,
  WSL_DOCKER_DESKTOP_GPU_COMPATIBILITY_REMOVAL_CONDITION,
  wslDockerDesktopGpuCompatibilityAction,
  wslDockerDesktopGpuCompatibilityRemediationLines,
  wslDockerDesktopGpuProofTimeoutMs,
} from "./wsl-docker-desktop-gpu";

describe("WSL Docker Desktop GPU compatibility helpers", () => {
  it("only matches Docker Desktop-backed WSL host assessments", () => {
    expect(isWslDockerDesktopRuntime({ isWsl: true, runtime: "docker-desktop" })).toBe(true);
    expect(isWslDockerDesktopRuntime({ isWsl: true, runtime: "docker" })).toBe(false);
    expect(isWslDockerDesktopRuntime({ isWsl: false, runtime: "docker-desktop" })).toBe(false);
  });

  it("detects Docker Desktop status only after WSL detection succeeds", () => {
    const dockerInfoFormat = vi.fn(() => '"Docker Desktop"');
    expect(
      detectWslDockerDesktopStatus({
        platform: "linux",
        env: { WSL_DISTRO_NAME: "Ubuntu" },
        dockerInfoFormat,
      }),
    ).toBe("docker-desktop");
    expect(dockerInfoFormat).toHaveBeenCalledWith(
      "{{json .OperatingSystem}}",
      expect.objectContaining({ ignoreError: true }),
    );

    expect(
      detectWslDockerDesktopStatus({
        platform: "linux",
        env: {},
        release: "6.8.0-generic",
        procVersion: "Linux version 6.8.0-generic",
        dockerInfoFormat: vi.fn(() => '"Docker Desktop"'),
      }),
    ).toBe("not-docker-desktop");
  });

  it("centralizes non-blocking Docker --gpus remediation and its removal condition", () => {
    const action = wslDockerDesktopGpuCompatibilityAction();
    expect(action.kind).toBe("info");
    expect(action.blocking).toBe(false);
    expect(action.reason).toContain("--gpus");
    expect(action.commands.join("\n")).not.toContain("nvidia-ctk");

    expect(wslDockerDesktopGpuCompatibilityRemediationLines("docker-desktop")?.join("\n")).toContain(
      "Docker --gpus compatibility",
    );
    expect(wslDockerDesktopGpuCompatibilityRemediationLines("unknown")?.join("\n")).toContain(
      "could not determine whether Docker is Docker Desktop",
    );
    expect(wslDockerDesktopGpuCompatibilityRemediationLines("not-docker-desktop")).toBeNull();
    expect(WSL_DOCKER_DESKTOP_GPU_COMPATIBILITY_REMOVAL_CONDITION).toContain("Remove");
  });
});

describe("createArm64WslDockerDesktopGpuProver (#4565)", () => {
  const passingProof = { passed: true, timedOut: false, exitCode: 0, diagnostic: "" };

  it("returns null on non-ARM64 hosts without running the proof", () => {
    const runProof = vi.fn(() => passingProof);
    const prover = createArm64WslDockerDesktopGpuProver({
      platform: "linux",
      arch: "x64",
      detectWslDockerDesktopStatus: () => "docker-desktop",
      runProof,
      log: () => undefined,
    });
    expect(prover(["JMJWOA-Generic-GPU"])).toBeNull();
    expect(runProof).not.toHaveBeenCalled();
  });

  it("returns null when the host is not Docker Desktop-backed WSL", () => {
    const runProof = vi.fn(() => passingProof);
    const prover = createArm64WslDockerDesktopGpuProver({
      platform: "linux",
      arch: "arm64",
      detectWslDockerDesktopStatus: () => "not-docker-desktop",
      runProof,
      log: () => undefined,
    });
    expect(prover(["JMJWOA-Generic-GPU"])).toBeNull();
    expect(runProof).not.toHaveBeenCalled();
  });

  it("runs the bounded proof and reports the result on ARM64 Docker Desktop WSL", () => {
    const runProof = vi.fn((_argv: string[], _timeoutMs: number) => passingProof);
    const prover = createArm64WslDockerDesktopGpuProver({
      platform: "linux",
      arch: "arm64",
      detectWslDockerDesktopStatus: () => "docker-desktop",
      runProof,
      log: () => undefined,
    });
    expect(prover(["JMJWOA-Generic-GPU"])).toEqual(passingProof);
    expect(runProof).toHaveBeenCalledTimes(1);
    const argv = runProof.mock.calls[0]?.[0] ?? [];
    expect(argv[0]).toBe("docker");
    expect(argv).toContain("--gpus");
  });

  it("propagates a failing proof so detection stays fail-closed", () => {
    const failing = { passed: false, timedOut: false, exitCode: 1, diagnostic: "no CUDA device" };
    const prover = createArm64WslDockerDesktopGpuProver({
      platform: "linux",
      arch: "arm64",
      detectWslDockerDesktopStatus: () => "docker-desktop",
      runProof: () => failing,
      log: () => undefined,
    });
    expect(prover(["JMJWOA-Generic-GPU"])?.passed).toBe(false);
  });

  it("honors a positive NEMOCLAW_WSL_GPU_PROOF_TIMEOUT_MS override", () => {
    expect(wslDockerDesktopGpuProofTimeoutMs({ NEMOCLAW_WSL_GPU_PROOF_TIMEOUT_MS: "5000" })).toBe(5000);
    expect(wslDockerDesktopGpuProofTimeoutMs({})).toBeGreaterThan(0);
    expect(wslDockerDesktopGpuProofTimeoutMs({ NEMOCLAW_WSL_GPU_PROOF_TIMEOUT_MS: "-1" })).toBeGreaterThan(
      0,
    );
  });
});
