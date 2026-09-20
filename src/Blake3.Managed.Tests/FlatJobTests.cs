using System.Collections.Concurrent;
using Blake3.Managed.Internal;

namespace Blake3.Managed.Tests;

/// <summary>
/// <see cref="Blake3Tree.FlatJob"/> is the thread-pool fan-out shared by the one-shot tree and
/// <c>UpdateWithJoin</c>. Its items read pinned memory owned by the caller's frame, so returning
/// before every claimed item has settled is a use-after-free, not merely a lost result. None of
/// that is observable through a digest: a job that abandons its workers early still produces the
/// right hash on a good day, so these tests drive the job directly instead.
/// </summary>
public class FlatJobTests
{
    /// <summary>
    /// Records which items ran, how many were running at once, and how many are still running.
    /// </summary>
    private sealed class RecordingJob : Blake3Tree.FlatJob
    {
        private readonly int? _throwAt;
        private int _inFlight;

        public readonly ConcurrentBag<int> Processed = new();
        public int MaxInFlight;
        public int InFlightAtReturn => Volatile.Read(ref _inFlight);

        public RecordingJob(int items, int? throwAt = null)
            : base(items)
        {
            _throwAt = throwAt;
        }

        protected override void Process(int item)
        {
            int now = Interlocked.Increment(ref _inFlight);
            InterlockedMax(ref MaxInFlight, now);
            try
            {
                if (item == _throwAt)
                {
                    throw new InvalidOperationException($"unit {item} failed");
                }

                // Enough work that other units are plausibly still running when one throws.
                // The assertions below do not depend on that happening, only on the drain rule.
                Thread.SpinWait(2_000);
                Processed.Add(item);
            }
            finally
            {
                Interlocked.Decrement(ref _inFlight);
            }
        }

        private static void InterlockedMax(ref int target, int value)
        {
            int seen = Volatile.Read(ref target);
            while (value > seen)
            {
                int actual = Interlocked.CompareExchange(ref target, value, seen);
                if (actual == seen) return;
                seen = actual;
            }
        }
    }

    /// <summary>
    /// Runs the job off the test thread and fails rather than hangs if it never returns.
    /// A drain bug does not always surface as a wrong answer: abandoning claimed units leaves
    /// the completion count short of the total, so the job waits for workers that will never
    /// report, and the natural failure mode is a deadlock. The generous budget keeps this from
    /// being a timing assertion -- these jobs finish in microseconds.
    /// </summary>
    private static Exception? RunBounded(Blake3Tree.FlatJob job, int degree)
    {
        Exception? caught = null;
        var task = Task.Run(() =>
        {
            try
            {
                job.RunOnPool(degree);
            }
            catch (Exception ex)
            {
                caught = ex;
            }
        });

        Assert.True(task.Wait(TimeSpan.FromSeconds(10)),
            "RunOnPool did not return: it is waiting on units it abandoned.");

        return caught;
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(8)]
    [InlineData(64)]
    public void RunsEveryItemExactlyOnce(int items)
    {
        var job = new RecordingJob(items);

        Assert.Null(RunBounded(job, Environment.ProcessorCount));

        Assert.Equal(Enumerable.Range(0, items), job.Processed.OrderBy(i => i));
        Assert.Equal(0, job.InFlightAtReturn);
    }

    /// <summary>
    /// The drain rule. A worker that throws must not cut the job short: every other item still
    /// runs, and nothing is still running when the call returns. This fails if anyone
    /// "simplifies" the claim loop to return or rethrow on the first fault, which is exactly the
    /// tidy-up someone would make -- and which would let the caller's frame pop while a worker
    /// is still reading the pinned input.
    /// </summary>
    [Theory]
    [InlineData(64, 0)]
    [InlineData(64, 17)]
    [InlineData(64, 63)]
    [InlineData(2, 1)]
    public void DrainsEveryOtherItemAndWaitsForWorkers_WhenOneThrows(int items, int throwAt)
    {
        var job = new RecordingJob(items, throwAt);

        var thrown = Assert.IsType<InvalidOperationException>(
            RunBounded(job, Environment.ProcessorCount));

        Assert.Equal($"unit {throwAt} failed", thrown.Message);

        // No worker may still be running: the caller is about to release the memory items read.
        Assert.Equal(0, job.InFlightAtReturn);

        // Every item except the faulted one completed.
        var expected = Enumerable.Range(0, items).Where(i => i != throwAt);
        Assert.Equal(expected, job.Processed.OrderBy(i => i));
    }

    /// <summary>
    /// A single item, or a degree of one, queues no workers at all and runs inline. That path
    /// shares the drain and rethrow code with the fan-out, so it gets the same guarantees.
    /// </summary>
    [Fact]
    public void RunsInlineWithoutWorkers_WhenDegreeIsOne()
    {
        var job = new RecordingJob(16);

        Assert.Null(RunBounded(job, 1));

        Assert.Equal(Enumerable.Range(0, 16), job.Processed.OrderBy(i => i));
        Assert.Equal(1, job.MaxInFlight);
        Assert.Equal(0, job.InFlightAtReturn);
    }

    [Fact]
    public void RunsInlineWithoutWorkers_WhenDegreeIsOne_AndAnItemThrows()
    {
        var job = new RecordingJob(16, throwAt: 4);

        Assert.IsType<InvalidOperationException>(RunBounded(job, 1));

        Assert.Equal(Enumerable.Range(0, 16).Where(i => i != 4), job.Processed.OrderBy(i => i));
        Assert.Equal(0, job.InFlightAtReturn);
    }

    /// <summary>
    /// Blocks every item that is not running on the caller's own thread, so the caller reaches
    /// the wait while a worker is still inside Process.
    /// </summary>
    private sealed class GatedJob : Blake3Tree.FlatJob
    {
        private readonly int _callerThreadId;
        public readonly ManualResetEventSlim WorkerEntered = new(false);
        public readonly ManualResetEventSlim Release = new(false);

        public GatedJob(int items, int callerThreadId) : base(items)
        {
            _callerThreadId = callerThreadId;
        }

        protected override void Process(int item)
        {
            if (Environment.CurrentManagedThreadId == _callerThreadId)
            {
                // The caller must not be able to drain every item before a worker has started,
                // or it reaches the wait with nothing outstanding and the test proves nothing.
                // Bounded so a starved pool fails the assertion rather than hanging the run.
                WorkerEntered.Wait(TimeSpan.FromSeconds(10));
                return;
            }

            WorkerEntered.Set();
            Release.Wait();
        }
    }

    /// <summary>
    /// Interrupting the waiting caller must not release it while a worker is still running.
    /// `ManualResetEventSlim.Wait` throws `ThreadInterruptedException`, and if that escaped
    /// `RunOnPool` the caller would leave its `fixed` block and free the pooled buffer with a
    /// worker still reading both -- a use-after-free reached without any item misbehaving, and
    /// the one hole the other tests here cannot see, because nothing in them throws.
    /// </summary>
    [Fact]
    public void DoesNotReturnWhenTheWaitingCallerIsInterrupted()
    {
        GatedJob job = null!;
        Thread caller = null!;
        var started = new ManualResetEventSlim(false);

        caller = new Thread(() =>
        {
            job = new GatedJob(items: 8, callerThreadId: Environment.CurrentManagedThreadId);
            started.Set();
            job.RunOnPool(Environment.ProcessorCount);
        })
        { IsBackground = true };

        caller.Start();
        Assert.True(started.Wait(TimeSpan.FromSeconds(10)));
        Assert.True(job.WorkerEntered.Wait(TimeSpan.FromSeconds(10)),
            "no worker ever entered Process, so this test proves nothing");

        caller.Interrupt();

        // With the interrupt escaping the wait, the caller returns essentially instantly, so a
        // generous budget here is not a timing assertion in any meaningful sense.
        Assert.False(caller.Join(TimeSpan.FromSeconds(2)),
            "RunOnPool returned while a worker was still running: an interrupt escaped the wait.");

        job.Release.Set();
        Assert.True(caller.Join(TimeSpan.FromSeconds(30)), "RunOnPool never returned.");
    }

    /// <summary>
    /// Several faults must not lose the failure or surface something unrelated; exactly one of
    /// them is reported and the rest of the work still drains.
    /// </summary>
    private sealed class MultiFaultJob : Blake3Tree.FlatJob
    {
        private int _completed;
        public int Completed => Volatile.Read(ref _completed);

        public MultiFaultJob(int items) : base(items) { }

        protected override void Process(int item)
        {
            if (item % 3 == 0) throw new InvalidOperationException("boom");
            Interlocked.Increment(ref _completed);
        }
    }

    [Fact]
    public void ReportsOneFailureAndStillDrains_WhenManyItemsThrow()
    {
        const int items = 30;
        var job = new MultiFaultJob(items);

        Assert.IsType<InvalidOperationException>(RunBounded(job, Environment.ProcessorCount));

        // Items 0, 3, ... 27 throw: ten of the thirty. The other twenty must still run.
        Assert.Equal(20, job.Completed);
    }
}
