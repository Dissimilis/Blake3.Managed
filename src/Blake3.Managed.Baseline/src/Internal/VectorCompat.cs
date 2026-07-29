using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Blake3.Managed.Internal;

internal static class VectorCompat
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<T> Load<T>(ref T source) where T : struct
    {
#if NET7_0_OR_GREATER
        return Vector128.LoadUnsafe(ref source);
#else
        return Unsafe.ReadUnaligned<Vector128<T>>(ref Unsafe.As<T, byte>(ref source));
#endif
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<T> Load<T>(ref T source, nuint elementOffset) where T : struct
    {
#if NET7_0_OR_GREATER
        return Vector128.LoadUnsafe(ref source, elementOffset);
#else
        return Unsafe.ReadUnaligned<Vector128<T>>(ref Unsafe.As<T, byte>(ref Unsafe.Add(ref source, (nint)elementOffset)));
#endif
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store<T>(Vector128<T> vector, ref T destination) where T : struct
    {
#if NET7_0_OR_GREATER
        vector.StoreUnsafe(ref destination);
#else
        Unsafe.WriteUnaligned(ref Unsafe.As<T, byte>(ref destination), vector);
#endif
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store<T>(Vector128<T> vector, ref T destination, nuint elementOffset) where T : struct
    {
#if NET7_0_OR_GREATER
        vector.StoreUnsafe(ref destination, elementOffset);
#else
        Unsafe.WriteUnaligned(ref Unsafe.As<T, byte>(ref Unsafe.Add(ref destination, (nint)elementOffset)), vector);
#endif
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store<T>(Vector256<T> vector, ref T destination) where T : struct
    {
#if NET7_0_OR_GREATER
        vector.StoreUnsafe(ref destination);
#else
        Unsafe.WriteUnaligned(ref Unsafe.As<T, byte>(ref destination), vector);
#endif
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store<T>(Vector256<T> vector, ref T destination, nuint elementOffset) where T : struct
    {
#if NET7_0_OR_GREATER
        vector.StoreUnsafe(ref destination, elementOffset);
#else
        Unsafe.WriteUnaligned(ref Unsafe.As<T, byte>(ref Unsafe.Add(ref destination, (nint)elementOffset)), vector);
#endif
    }
}
