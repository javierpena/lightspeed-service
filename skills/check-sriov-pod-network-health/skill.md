---
name: check-sriov-pod-network-health
description: Run a complete health check of the network for an OpenShift pod that uses SR-IOV. Use when user wants to troubleshoot or check the health of a pod running workloads that use SR-IOV.
---


# Check SR-IOV pod network health

## When to Use

- Use this skill when you need to check the configuration of an OpenShift pod using SR-IOV, to achieve high performance
- This skill is helpful to verify the configuration of the OpenShift cluster, nodes and pod to obtain high SR-IOV network performance

## Rules

- Always use the MCP tools at your disposal
- NEVER try to run any commands outside of MCP tool calls
- You will focus the analysis on pod $ARGUMENTS[0] from namespace $ARGUMENTS[1]

## Step 1: Check OpenShift node configuration

1. Check the CPU usage of all reserved cores, as defined by the Performance Profile resource. Refer to `references/cpu-per-core.md` for detailed instructions.
2. Make sure the topology policy defined in the Performance Profile resource is either "single-numa-node" or "restricted".

## Step 2: Check pod configuration

1. Make sure the following annotations, including their required values, are included in the pod definition:
    - cpu-load-balancing.crio.io: disable
    - cpu-quota.crio.io: disable
    - irq-load-balancing.crio.io: disable
2. Make sure the the pod's CPU utilization is above 90%.
3. Make sure the pod's containers are not being throttled by the CFS.
4. Make sure the QoS class for the pod is Guaranteed.
5. Find the node CPUs assigned to the pod. Refer to `references/find-cpus-for-pod.md` for detailed instructions.

## Step 3: Check OpenShift node network configuration

1. Find the physical NICs used by the SR-IOV VFs associated to the pod. Refer to `references/find-physical-nic-for-sriov-pod.md` for detailed instructions.
2. Check the MTU for all physical NICs used by the SR-IOV VFs associated to the pod. They must be 1500 or higher.
3. Check the combined channels for the physical NICs used by the SR-IOV VFs associated to the pod. The number of combined channels must be at least 16 for each NIC.
4. Check the statistics for all physical NICs used by the SR-IOV VFs associated to the pod. You can use the query_ethtool tool to get that information. There should be a good balance in the values of tx_queue_*_packets and rx_queue_*_packets for each of the tx and rx queues.
5. Check the MTU for all SR-IOV VFs associated to the pod. They must be 1500 or higher.
6. Make sure there are no errors or packet drops shown for any physical NICs used by the SR-IOV VFs associated to the pod. Refer to `references/nic-errors-packet-drops.md` for detailed instructions.
7. Check for any drops or errors at the TCP and UDP layers on the node running the pod. Refer to `references/tcp-udp-layers-information.md` for detailed instructions.

## Step 4: Check low-level OpenShift node configuration

1. Check which processes are running on the node CPUs assigned to the pod. If there is any kernel process running on those CPUs, ensure it is a per-cpu kernel thread and not any other type of process.
2. Check the IRQs allowed to run on the node CPUs assigned to the pod. No IRQ related to a network driver should be allowed to run on the isolated CPUs used by the pod. It is ok to have those IRQs running on the system's reserved CPUs.
3. Check the kernel settings under /proc/sys/net are correct. Refer to `references/recommended-sriov-net-kernel-settings.md` for detailed information.
4. Check for softnet packet-drop errors or high time_squeeze values, which can indicate network contention on the node running the pod. Refer to `references/softnet.md` for detailed instructions.
5. Check for a high number of SMI received by the node's CPU. Refer to `references/smi.md` for detailed instructions.

## Step 5: Final report

Report status of each of the checks, providing a summary of the next steps.
