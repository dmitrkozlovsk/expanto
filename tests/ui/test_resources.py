import queue
from concurrent.futures import Future, ThreadPoolExecutor

from streamlit.testing.v1 import AppTest


def thread_pool_app(max_workers: int | None = None, num_tasks: int = 10) -> None:
    """Test Streamlit app for ThreadPoolExecutor resource."""
    import queue
    import time
    from threading import get_ident

    import streamlit as st

    from src.ui.resources import get_thread_pool_executor

    def task(q: queue.Queue) -> None:
        """A simple task that puts the thread identifier into a queue."""
        time.sleep(0.01)  # Simulate some work
        q.put(get_ident())

    # Use a queue to collect results from threads
    if "results_queue" not in st.session_state:
        st.session_state.results_queue = queue.Queue()
    results_q = st.session_state.results_queue

    pool = get_thread_pool_executor(max_workers=max_workers)
    st.session_state.pool = pool

    # Submit tasks and store futures in session state
    if "futures" not in st.session_state:
        st.session_state.futures = []

        def task_func():
            task(results_q)

        for _ in range(num_tasks):
            future = pool.submit(task_func)
            st.session_state.futures.append(future)


def test_get_thread_pool_executor_creation_and_singleton():
    """Verify that the thread pool is created and is a singleton."""
    at = AppTest.from_function(thread_pool_app)
    at.run()

    # 1. Verify pool creation
    assert "pool" in at.session_state
    assert isinstance(at.session_state.pool, ThreadPoolExecutor)
    pool_id_first_run = id(at.session_state.pool)

    # 2. Verify singleton behavior on rerun
    at.run()
    assert id(at.session_state.pool) == pool_id_first_run
    at.run()
    assert id(at.session_state.pool) == pool_id_first_run


def test_thread_pool_task_execution():
    """Verify that tasks submitted to the pool are executed."""
    max_workers = 8
    num_tasks = 10
    at = AppTest.from_function(thread_pool_app, args=(max_workers, num_tasks))
    at.run()

    # 3. Verify task execution
    futures: list[Future] = at.session_state.futures
    # Wait for all futures to complete
    for future in futures:
        future.result()  # This will block until the future is done

    results_queue: queue.Queue = at.session_state.results_queue
    assert results_queue.qsize() == num_tasks

    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    assert len(set(results)) == max_workers
