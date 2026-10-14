using System.Runtime.Intrinsics.Arm;

namespace Blake3.Managed.Internal;

/// <summary>Explicit CPU tuning; instruction support alone does not identify a CPU model.</summary>
internal static class ArmTuning
{
    // Read once. Applications must set this before the first use of the library.
    internal static readonly bool CortexA510 = AdvSimd.Arm64.IsSupported
        && AppContext.TryGetSwitch("Blake3.Managed.TuneForCortexA510", out bool enabled) && enabled;

    internal static bool UseScalarNeonTails
    {
        get
        {
#if NET10_0_OR_GREATER
#pragma warning disable SYSLIB5003
            if (Sve2.IsSupported) return false;
#pragma warning restore SYSLIB5003
#endif
            return CortexA510;
        }
    }
}
