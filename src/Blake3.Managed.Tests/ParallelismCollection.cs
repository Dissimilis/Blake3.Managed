namespace Blake3.Managed.Tests;

/// <summary>
/// Classes that either write <see cref="Hasher.MaxDegreeOfParallelism"/> or depend on the
/// parallel dispatch actually running.
/// </summary>
/// <remarks>
/// That setting is process-wide, and xUnit runs test classes in parallel by default. So a class
/// that sets it to 1 to check the serial path silently forces every other class running at that
/// moment onto the serial path too: while it is 1, nothing enters the thread-pool tree or the
/// load-gated band at all. A bug in the parallel fold would then pass or fail depending on test
/// scheduling, which is the worst property a suite can have -- it does not fail, it stops
/// asking the question, and the run still says green.
///
/// The same applies to <c>FanOutGateTests</c>, which reads the process-wide in-flight counter:
/// any other class hashing a mid-size input at that moment makes the reading meaningless.
///
/// Everything that touches either piece of shared state joins this collection, which xUnit will
/// not schedule alongside anything else.
/// </remarks>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class ParallelismCollection
{
    public const string Name = "parallel dispatch and its process-wide state";
}
