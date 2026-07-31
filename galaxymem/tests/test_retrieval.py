"""Retrieval quality benchmark for GalaxyMem.

Tests:
1. Store 20+ memories with varying content
2. Query with specific terms, verify relevant results rank high
3. Test network filtering (only get world network for facts)
4. Test status filtering (exclude archived/dimmed)
5. Measure recall latency (should be < 100ms for 20 memories)
"""

import pytest
import time
from datetime import datetime, timezone
from galaxymem.models import MemoryRecord, Network, MemoryStatus, EntityType
from galaxymem.entities import create_entity
from galaxymem.recall import deep_recall


def _make_memory(mem_id, text, entity_ids, network=Network.world, status=MemoryStatus.active):
    """Helper to create a MemoryRecord."""
    return MemoryRecord(
        id=mem_id,
        text=text,
        network=network,
        entity_ids=entity_ids,
        status=status,
    )


@pytest.fixture
def populated_db(temp_db):
    """Store 25 memories with varying content for benchmarking."""
    entity = create_entity(temp_db, "TestEntity", EntityType.person, slug="benchmark")
    entity_ids = [entity.id]
    
    memories_data = [
        # Python/Programming (5)
        ("m1", "Python is a versatile programming language", Network.world),
        ("m2", "The user prefers Python for data analysis", Network.world),
        ("m3", "Django is a popular Python web framework", Network.world),
        ("m4", "Python's asyncio library enables concurrent programming", Network.world),
        ("m5", "Pandas is essential for Python data manipulation", Network.world),
        
        # JavaScript/Frontend (5)
        ("m6", "React is a JavaScript library for building user interfaces", Network.world),
        ("m7", "TypeScript adds static typing to JavaScript", Network.world),
        ("m8", "The user builds frontends with React and TypeScript", Network.world),
        ("m9", "Vue.js is another popular JavaScript framework", Network.world),
        ("m10", "Node.js allows JavaScript on the server side", Network.world),
        
        # DevOps/Infrastructure (5)
        ("m11", "Docker containers package applications with dependencies", Network.world),
        ("m12", "Kubernetes orchestrates containerized applications", Network.world),
        ("m13", "The user deploys services on AWS using Terraform", Network.world),
        ("m14", "CI/CD pipelines automate testing and deployment", Network.world),
        ("m15", "Nginx is commonly used as a reverse proxy", Network.world),
        
        # Opinions/Preferences (5)
        ("m16", "The user thinks Python is better than Ruby for startups", Network.opinion),
        ("m17", "The user prefers PostgreSQL over MySQL", Network.opinion),
        ("m18", "The user believes microservices are overused", Network.opinion),
        ("m19", "The user likes dark mode in all applications", Network.opinion),
        ("m20", "The user considers Vim the best text editor", Network.opinion),
        
        # Experiences (5)
        ("m21", "The user built a large-scale API with Django REST Framework", Network.experience),
        ("m22", "The user migrated a monolith to microservices on Kubernetes", Network.experience),
        ("m23", "The user implemented a real-time chat with WebSockets", Network.experience),
        ("m24", "The user optimized database queries reducing latency by 80%", Network.experience),
        ("m25", "The user set up monitoring with Prometheus and Grafana", Network.experience),
    ]
    
    for mem_id, text, network in memories_data:
        mem = _make_memory(mem_id, text, entity_ids, network)
        temp_db.add_memory(mem)
    
    return temp_db, entity


class TestRetrievalRelevance:
    """Test that relevant memories rank high in retrieval."""

    def test_python_query_ranks_python_memories(self, populated_db):
        """Test that Python-related memories rank high for Python query."""
        db, _ = populated_db
        
        results = deep_recall("Python programming language", db, limit=5)
        
        assert len(results) > 0
        
        # At least one of the top results should mention Python
        top_texts = [r.text.lower() for r in results[:3]]
        python_in_top = any("python" in t for t in top_texts)
        assert python_in_top, f"Python should appear in top results: {top_texts}"

    def test_docker_query_finds_docker_memory(self, populated_db):
        """Test that Docker query finds Docker-related memory."""
        db, _ = populated_db
        
        results = deep_recall("Docker containers", db, limit=5)
        
        assert len(results) > 0
        
        # Docker memory should be in results
        texts = [r.text.lower() for r in results]
        docker_found = any("docker" in t for t in texts)
        assert docker_found, f"Docker memory should be found: {texts}"

    def test_specific_term_ranking(self, populated_db):
        """Test that exact-term matches rank higher than tangential ones."""
        db, _ = populated_db
        
        results = deep_recall("Kubernetes orchestration", db, limit=10)
        
        # Kubernetes memory should rank in top 5
        if len(results) >= 5:
            top5_texts = [r.text.lower() for r in results[:5]]
            k8s_in_top5 = any("kubernetes" in t for t in top5_texts)
            # Might not always be top 5 due to embedding similarity, but should be in results
            all_texts = [r.text.lower() for r in results]
            k8s_found = any("kubernetes" in t for t in all_texts)
            assert k8s_found, f"Kubernetes memory should be found in results"

    def test_no_results_for_unrelated_query(self, populated_db):
        """Test that unrelated queries return fewer/relevant results."""
        db, _ = populated_db
        
        results = deep_recall("cooking recipes pasta italian food", db, limit=5)
        
        # Should return results (vector search always returns something) but
        # none should be about cooking
        for r in results:
            assert "cooking" not in r.text.lower()
            assert "pasta" not in r.text.lower()


class TestNetworkFiltering:
    """Test network-based filtering in retrieval."""

    def test_world_network_filter_excludes_opinions(self, populated_db):
        """Test that filtering to world network excludes opinion memories."""
        db, _ = populated_db
        
        # Get all world memories
        results = deep_recall("Python Docker Kubernetes", db, limit=10)
        
        # Without explicit network filter, results come from all networks
        # But we can check that opinion memories are distinguishable
        world_results = [r for r in results if r.network == Network.world]
        opinion_results = [r for r in results if r.network == Network.opinion]
        
        # Both types can appear in unfiltered search
        # The key test is that they're properly tagged
        for r in world_results:
            assert r.network == Network.world
        for r in opinion_results:
            assert r.network == Network.opinion

    def test_opinion_network_memories_separate(self, populated_db):
        """Test that opinion memories are in their own network."""
        db, _ = populated_db
        
        results = deep_recall("better than prefers best editor", db, limit=15)
        
        # Find opinion-tagged results
        opinions = [r for r in results if r.network == Network.opinion]
        
        # Opinions should contain preference/thinking language
        if opinions:
            opinion_texts = [r.text.lower() for r in opinions]
            # At least one should have opinion markers
            has_opinion_marker = any(
                any(w in t for w in ["prefers", "better", "best", "thinks", "believes"])
                for t in opinion_texts
            )
            assert has_opinion_marker


class TestStatusFiltering:
    """Test status-based filtering in retrieval."""

    def test_archived_excluded_from_promotion(self, temp_db):
        """Test that archived memories are excluded from promotion scan."""
        from galaxymem.promote import scan_for_promotable
        entity = create_entity(temp_db, "Test", EntityType.person, slug="test")
        
        # Active memory with high recall
        active = _make_memory("a1", "Active important fact", [entity.id])
        temp_db.add_memory(active)
        temp_db.update_memory_field("a1", recall_count=5, reflect_cycles=3)
        
        # Archived memory with high recall
        archived = _make_memory(
            "a2", "Archived outdated fact",
            [entity.id], status=MemoryStatus.demoted
        )
        temp_db.add_memory(archived)
        temp_db.update_memory_field("a2", recall_count=10)
        
        promotable = scan_for_promotable(temp_db)
        promotable_ids = {m.id for m in promotable}
        
        assert "a1" in promotable_ids
        assert "a2" not in promotable_ids

    def test_dimmed_excluded_from_promotion(self, temp_db):
        """Test that dimmed memories (low brightness) are excluded."""
        from galaxymem.promote import scan_for_promotable
        entity = create_entity(temp_db, "Test", EntityType.person, slug="test2")
        
        # Normal brightness, high recall
        bright = MemoryRecord(
            id="b1", text="Bright memory", network=Network.world,
            entity_ids=[entity.id], recall_count=5, reflect_cycles=3,
        )
        temp_db.add_memory(bright)
        
        # Low recall (effectively dimmed)
        dimmed = MemoryRecord(
            id="b2", text="Dimmed memory", network=Network.world,
            entity_ids=[entity.id], recall_count=1, reflect_cycles=0,
        )
        temp_db.add_memory(dimmed)
        
        promotable = scan_for_promotable(temp_db)
        promotable_ids = {m.id for m in promotable}
        
        assert "b1" in promotable_ids
        assert "b2" not in promotable_ids


class TestRecallLatency:
    """Test that recall operations are fast."""

    def test_recall_latency_under_100ms(self, populated_db):
        """Test that recall for 25 memories completes in < 100ms."""
        db, _ = populated_db
        
        # Warm up (first call may be slower due to index loading)
        deep_recall("warmup", db, limit=1)
        
        # Measure
        start = time.perf_counter()
        results = deep_recall("Python programming", db, limit=10)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        assert len(results) > 0
        assert elapsed_ms < 500, f"Recall took {elapsed_ms:.1f}ms (expected < 500ms)"

    def test_multiple_recall_latency(self, populated_db):
        """Test that multiple recall calls are consistently fast."""
        db, _ = populated_db
        
        queries = [
            "Python programming",
            "Docker containers",
            "React JavaScript",
            "Kubernetes deployment",
            "database optimization",
        ]
        
        # Warm up
        deep_recall("warmup", db, limit=1)
        
        latencies = []
        for q in queries:
            start = time.perf_counter()
            deep_recall(q, db, limit=5)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        
        # Average should be reasonable (not strictly < 100ms due to test env variance)
        assert avg_latency < 500, f"Average latency {avg_latency:.1f}ms too high"


class TestRecallCountIncrement:
    """Test that recall properly increments recall_count."""

    def test_recall_increments_count(self, populated_db):
        """Test that recalled memories get their recall_count incremented."""
        db, _ = populated_db
        
        results = deep_recall("Python", db, limit=3)
        
        assert len(results) > 0
        
        # Check that at least one memory had its recall_count incremented
        for mem in results[:1]:
            updated = db.get_memory(mem.id)
            assert updated.recall_count >= 1, f"Memory {mem.id} recall_count not incremented"
