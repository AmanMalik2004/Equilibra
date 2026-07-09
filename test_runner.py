import unittest
import sys
import io
import os
from datetime import datetime

OUTPUT_FILE = "test_results.txt"

class TeeStream:
    """Write to both a stream and a file simultaneously."""
    def __init__(self, stream, file_obj):
        self.stream = stream
        self.file   = file_obj
    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
    def flush(self):
        self.stream.flush()
        self.file.flush()

from servers.server import Server
from algorithms.round_robin import RoundRobin
from algorithms.least_connection import LeastConnections
from ml_model.predictor import AILoadBalancer
from ml_model.train import train_model
from traffic.generator import generate_requests
from metrics.tracker import MetricTracker

if not os.path.exists("ml_model/model.pkl"):
    train_model()

class TestServer(unittest.TestCase):

    def test_initial_load_is_zero(self):
        """Server should start with zero load"""
        s = Server("SRV-1")
        self.assertEqual(s.current_load, 0)

    def test_initial_requests_zero(self):
        """Server should start with zero total requests"""
        s = Server("SRV-1")
        self.assertEqual(s.total_requests, 0)

    def test_handle_request_returns_positive_time(self):
        """Response time must always be positive"""
        s = Server("SRV-1")
        rt = s.handle_request(1)
        self.assertGreater(rt, 0)

    def test_handle_request_increments_total(self):
        """Each request should increment total_requests"""
        s = Server("SRV-1")
        s.handle_request(1)
        s.handle_request(2)
        self.assertEqual(s.total_requests, 2)

    def test_avg_response_time_no_requests(self):
        """Avg response time should be 0 if no requests handled"""
        s = Server("SRV-1")
        self.assertEqual(s.avg_response_time(), 0)

    def test_avg_response_time_positive_after_requests(self):
        """Avg response time should be positive after handling requests"""
        s = Server("SRV-1")
        s.handle_request(1)
        self.assertGreater(s.avg_response_time(), 0)

    def test_utilization_equals_total_requests(self):
        """Utilization should match total requests handled"""
        s = Server("SRV-1")
        s.handle_request(1)
        s.handle_request(2)
        s.handle_request(3)
        self.assertEqual(s.utilization(), 3)

    def test_reset_clears_all(self):
        """Reset should zero out all stats"""
        s = Server("SRV-1")
        s.handle_request(1)
        s.reset()
        self.assertEqual(s.current_load, 0)
        self.assertEqual(s.total_requests, 0)
        self.assertEqual(s.total_response_time, 0)

    def test_busier_server_slower(self):
        """Higher load should generally produce higher response time"""
        s1 = Server("SRV-1")
        s2 = Server("SRV-2")
        s2.current_load = 20
        rt1 = s1.handle_request(1)
        rt2 = s2.handle_request(2)
        self.assertGreater(rt2, rt1)


class TestRoundRobin(unittest.TestCase):

    def setUp(self):
        self.servers = [Server("SRV-1"), Server("SRV-2"), Server("SRV-3")]
        self.rr = RoundRobin(self.servers)

    def test_first_pick_is_server_0(self):
        """First request should go to Server 0"""
        s = self.rr.get_server()
        self.assertEqual(s.name, "SRV-1")

    def test_second_pick_is_server_1(self):
        """Second request should go to Server 1"""
        self.rr.get_server()
        s = self.rr.get_server()
        self.assertEqual(s.name, "SRV-2")

    def test_third_pick_is_server_2(self):
        """Third request should go to Server 2"""
        self.rr.get_server()
        self.rr.get_server()
        s = self.rr.get_server()
        self.assertEqual(s.name, "SRV-3")

    def test_wraps_around_to_server_0(self):
        """After 3 picks it should wrap back to Server 0"""
        for _ in range(3):
            self.rr.get_server()
        s = self.rr.get_server()
        self.assertEqual(s.name, "SRV-1")

    def test_even_distribution_over_300_requests(self):
        """Over 300 requests each server should get exactly 100"""
        counts = {"SRV-1": 0, "SRV-2": 0, "SRV-3": 0}
        for _ in range(300):
            s = self.rr.get_server()
            counts[s.name] += 1
        self.assertEqual(counts["SRV-1"], 100)
        self.assertEqual(counts["SRV-2"], 100)
        self.assertEqual(counts["SRV-3"], 100)

    def test_returns_server_object(self):
        """get_server should return a Server instance"""
        s = self.rr.get_server()
        self.assertIsInstance(s, Server)


class TestLeastConnections(unittest.TestCase):

    def setUp(self):
        self.servers = [Server("SRV-1"), Server("SRV-2"), Server("SRV-3")]
        self.lc = LeastConnections(self.servers)

    def test_picks_server_with_least_load(self):
        """Should pick the server with lowest current_load"""
        self.servers[0].current_load = 5
        self.servers[1].current_load = 2
        self.servers[2].current_load = 8
        s = self.lc.get_server()
        self.assertEqual(s.name, "SRV-2")

    def test_picks_first_when_all_equal(self):
        """When all loads equal, should pick first server"""
        s = self.lc.get_server()
        self.assertEqual(s.name, "SRV-1")

    def test_avoids_overloaded_server(self):
        """Should never pick the server with highest load"""
        self.servers[2].current_load = 100
        for _ in range(10):
            s = self.lc.get_server()
            self.assertNotEqual(s.name, "SRV-3")

    def test_returns_server_object(self):
        """get_server should return a Server instance"""
        s = self.lc.get_server()
        self.assertIsInstance(s, Server)

    def test_adapts_to_changing_loads(self):
        self.servers[0].current_load = 3
        self.servers[1].current_load = 1
        self.servers[2].current_load = 5
        s1 = self.lc.get_server()
        self.assertEqual(s1.name, "SRV-2")
        self.servers[1].current_load = 10
        s2 = self.lc.get_server()
        self.assertNotEqual(s2.name, "SRV-2")


class TestAILoadBalancer(unittest.TestCase):

    def setUp(self):
        self.servers = [Server("SRV-1"), Server("SRV-2"), Server("SRV-3")]
        self.ai = AILoadBalancer(self.servers)

    def test_returns_server_object(self):
        """get_server should return a Server instance"""
        s = self.ai.get_server()
        self.assertIsInstance(s, Server)

    def test_returns_valid_server(self):
        """Should return one of the 3 known servers"""
        s = self.ai.get_server()
        self.assertIn(s.name, ["SRV-1", "SRV-2", "SRV-3"])

    def test_avoids_overloaded_server(self):
        """AI should generally avoid heavily loaded servers"""
        self.servers[0].current_load = 50
        self.servers[1].current_load = 50
        self.servers[2].current_load = 0
        picks = [self.ai.get_server().name for _ in range(10)]
        self.assertGreater(picks.count("SRV-3"), picks.count("SRV-1"))

    def test_model_file_exists(self):
        """Model pickle file should exist after training"""
        self.assertTrue(os.path.exists("ml_model/model.pkl"))

    def test_consistent_output_type(self):
        """Multiple calls should always return a Server"""
        for _ in range(20):
            s = self.ai.get_server()
            self.assertIsInstance(s, Server)


class TestMetricsTracker(unittest.TestCase):

    def test_initial_avg_is_zero(self):
        """Average should be 0 with no records"""
        t = MetricTracker()
        self.assertEqual(t.average_response_time(), 0)

    def test_total_requests_starts_zero(self):
        """Total requests should start at 0"""
        t = MetricTracker()
        self.assertEqual(t.total_requests(), 0)

    def test_record_increments_count(self):
        """Recording a time should increment total_requests"""
        t = MetricTracker()
        t.record(0.5)
        self.assertEqual(t.total_requests(), 1)

    def test_average_correct(self):
        """Average should be mathematically correct"""
        t = MetricTracker()
        t.record(0.2)
        t.record(0.4)
        t.record(0.6)
        self.assertAlmostEqual(t.average_response_time(), 0.4, places=5)

    def test_record_multiple(self):
        """Should correctly count multiple records"""
        t = MetricTracker()
        for i in range(50):
            t.record(0.1 * i)
        self.assertEqual(t.total_requests(), 50)

    def test_single_record_avg_equals_value(self):
        """Average of one record should equal that record"""
        t = MetricTracker()
        t.record(0.35)
        self.assertAlmostEqual(t.average_response_time(), 0.35, places=5)


class TestTrafficGenerator(unittest.TestCase):

    def test_generates_correct_count(self):
        reqs = generate_requests(100)
        self.assertEqual(len(reqs), 100)

    def test_each_request_has_id(self):
        reqs = generate_requests(10)
        for r in reqs:
            self.assertIn("id", r)

    def test_each_request_has_arrival_time(self):
        reqs = generate_requests(10)
        for r in reqs:
            self.assertIn("arrival time", r)

    def test_each_request_has_duration(self):
        reqs = generate_requests(10)
        for r in reqs:
            self.assertIn("duration", r)

    def test_ids_are_sequential(self):
        reqs = generate_requests(5)
        ids = [r["id"] for r in reqs]
        self.assertEqual(ids, [1, 2, 3, 4, 5])

    def test_bva_minimum_30(self):
        reqs = generate_requests(30)
        self.assertEqual(len(reqs), 30)

    def test_bva_just_above_minimum_31(self):
        reqs = generate_requests(31)
        self.assertEqual(len(reqs), 31)

    def test_bva_just_below_maximum_299(self):
        reqs = generate_requests(299)
        self.assertEqual(len(reqs), 299)

    def test_bva_maximum_300(self):
        reqs = generate_requests(300)
        self.assertEqual(len(reqs), 300)

    def test_bva_zero_requests(self):
        reqs = generate_requests(0)
        self.assertEqual(len(reqs), 0)

    def test_bva_one_request(self):
        reqs = generate_requests(1)
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0]["id"], 1)

    def test_ec_valid_small(self):
        reqs = generate_requests(50)
        self.assertEqual(len(reqs), 50)

    def test_ec_valid_medium(self):
        reqs = generate_requests(150)
        self.assertEqual(len(reqs), 150)

    def test_ec_valid_large(self):
        reqs = generate_requests(250)
        self.assertEqual(len(reqs), 250)


class TestIntegration(unittest.TestCase):

    def _run(self, balancer, servers, n=50):
        requests = generate_requests(n)
        tracker = MetricTracker()
        for req in requests:
            server = balancer.get_server()
            rt = server.handle_request(req["id"])
            tracker.record(rt)
        return tracker, servers

    def test_rr_full_pipeline(self):
        """Integration: Round Robin handles all requests"""
        servers = [Server("SRV-1"), Server("SRV-2"), Server("SRV-3")]
        tracker, _ = self._run(RoundRobin(servers), servers)
        self.assertEqual(tracker.total_requests(), 50)
        self.assertGreater(tracker.average_response_time(), 0)

    def test_lc_full_pipeline(self):
        """Integration: Least Connections handles all requests"""
        servers = [Server("SRV-1"), Server("SRV-2"), Server("SRV-3")]
        tracker, _ = self._run(LeastConnections(servers), servers)
        self.assertEqual(tracker.total_requests(), 50)
        self.assertGreater(tracker.average_response_time(), 0)

    def test_ai_full_pipeline(self):
        """Integration: AI Balancer handles all requests"""
        servers = [Server("SRV-1"), Server("SRV-2"), Server("SRV-3")]
        tracker, _ = self._run(AILoadBalancer(servers), servers)
        self.assertEqual(tracker.total_requests(), 50)
        self.assertGreater(tracker.average_response_time(), 0)

    def test_total_requests_distributed_across_servers(self):
        """Integration: total requests across servers should equal n"""
        servers = [Server("SRV-1"), Server("SRV-2"), Server("SRV-3")]
        self._run(RoundRobin(servers), servers, n=90)
        total = sum(s.utilization() for s in servers)
        self.assertEqual(total, 90)

    def test_all_algos_produce_positive_avg(self):
        """Integration: all algorithms should produce positive avg response time"""
        for AlgoClass in [RoundRobin, LeastConnections, AILoadBalancer]:
            servers = [Server("SRV-1"), Server("SRV-2"), Server("SRV-3")]
            tracker, _ = self._run(AlgoClass(servers), servers)
            self.assertGreater(tracker.average_response_time(), 0)


if __name__ == "__main__":
    with open(OUTPUT_FILE, "w") as f:

        header = (
            "=" * 70 + "\n"
            f"  TEST RUN  —  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}\n"
            "=" * 70 + "\n\n"
        )
        f.write(header)
        sys.stdout.write(header)

        tee = TeeStream(sys.stderr, f)          
        loader = unittest.TestLoader()
        suite  = loader.loadTestsFromModule(
            sys.modules[__name__]
        )
        runner = unittest.TextTestRunner(
            stream    = tee,
            verbosity = 2,
        )
        result = runner.run(suite)

        total   = result.testsRun
        passed  = total - len(result.failures) - len(result.errors)
        failed  = len(result.failures)
        errors  = len(result.errors)
        skipped = len(result.skipped)

        footer = (
            "\n" + "=" * 70 + "\n"
            "SUMMARY\n"
            + "-" * 70 + "\n"
            f"  Total   : {total}\n"
            f"  Passed  : {passed}\n"
            f"  Failed  : {failed}\n"
            f"  Errors  : {errors}\n"
            f"  Skipped : {skipped}\n"
            + "=" * 70 + "\n"
        )
        f.write(footer)
        sys.stdout.write(footer)

        print(f"\n[✓] Results saved to '{OUTPUT_FILE}'")

    sys.exit(0 if result.wasSuccessful() else 1)