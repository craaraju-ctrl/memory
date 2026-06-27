//! # Knowledge Graph — Relationship Tracking with Recursive Traversal
//!
//! Provides a relationship graph over memory records, enabling:
//! - Directed edge CRUD (source → target with typed relations)
//! - Recursive BFS traversal for multi-hop reasoning
//! - Relationship search by type
//! - Basic community detection via connectivity analysis

use std::collections::{HashMap, HashSet, VecDeque};

use crate::store::MemoryStore;
use crate::types::{GraphEdge, GraphTraversalResult, MemoryRecord};

/// Knowledge graph operating over the SQLite-backed edge store.
pub struct KnowledgeGraph {
    store: MemoryStore,
}

impl KnowledgeGraph {
    pub fn new(store: MemoryStore) -> Self {
        Self { store }
    }

    /// Add a directed edge between two records with a typed relationship.
    pub fn add_edge(
        &self,
        source_id: &str,
        target_id: &str,
        relation_type: &str,
        weight: f64,
    ) -> rusqlite::Result<String> {
        self.store.add_edge(source_id, target_id, relation_type, weight)
    }

    /// Add a bidirectional (undirected) edge.
    pub fn add_bidirectional_edge(
        &self,
        id_a: &str,
        id_b: &str,
        relation_type: &str,
        weight: f64,
    ) -> rusqlite::Result<(String, String)> {
        let e1 = self.store.add_edge(id_a, id_b, relation_type, weight)?;
        let e2 = self.store.add_edge(id_b, id_a, relation_type, weight)?;
        Ok((e1, e2))
    }

    /// Get all edges incident to a record.
    pub fn get_edges(&self, record_id: &str) -> rusqlite::Result<Vec<GraphEdge>> {
        self.store.get_edges(record_id)
    }

    /// Find the degree (number of connections) of a record in the graph.
    pub fn degree(&self, record_id: &str) -> rusqlite::Result<usize> {
        let edges = self.get_edges(record_id)?;
        // Count unique neighbors (handle bidirectional)
        let neighbors: HashSet<&str> = edges
            .iter()
            .flat_map(|e| {
                if e.source_id == record_id {
                    Some(e.target_id.as_str())
                } else {
                    Some(e.source_id.as_str())
                }
            })
            .collect();
        Ok(neighbors.len())
    }

    /// BFS traversal from a starting node up to max_depth.
    pub fn bfs(&self, start_id: &str, max_depth: u32, relation_filter: Option<&str>) -> rusqlite::Result<Vec<GraphTraversalResult>> {
        self.store.graph_bfs(start_id, max_depth, relation_filter)
    }

    /// Get related records with their relationship context.
    pub fn get_related(
        &self,
        record_id: &str,
        relation_type: Option<&str>,
        max_depth: u32,
    ) -> rusqlite::Result<Vec<(MemoryRecord, u32, String, f64)>> {
        self.store.get_related_records(record_id, relation_type, max_depth)
    }

    /// Find a path between two records (shortest path via BFS).
    pub fn find_path(&self, from_id: &str, to_id: &str, max_depth: u32) -> rusqlite::Result<Option<GraphTraversalResult>> {
        let results = self.bfs(from_id, max_depth, None)?;
        Ok(results.into_iter().find(|r| r.node_id == to_id))
    }

    /// Detect communities using connected-component analysis.
    /// Returns a map of component ID → list of record IDs in that component.
    pub fn detect_communities(&self, max_records: usize) -> rusqlite::Result<HashMap<u32, Vec<String>>> {
        // Get all edges using direct row iteration
        let mut edges: Vec<(String, String)> = Vec::new();
        {
            let conn = self.store.conn.lock().map_err(|e| {
                rusqlite::Error::InvalidParameterName(format!("Mutex poisoned: {}", e))
            })?;
            let mut stmt = conn.prepare(
                "SELECT DISTINCT source_id, target_id FROM graph_edges",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let src: String = row.get(0)?;
                let tgt: String = row.get(1)?;
                edges.push((src, tgt));
            }
        }

        // Build adjacency list
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for (src, tgt) in &edges {
            adj.entry(src.clone()).or_default().push(tgt.clone());
            adj.entry(tgt.clone()).or_default().push(src.clone());
        }

        // BFS component detection
        let mut visited: HashSet<String> = HashSet::new();
        let mut communities: HashMap<u32, Vec<String>> = HashMap::new();
        let mut component_id = 0u32;

        for node in adj.keys() {
            if visited.contains(node) {
                continue;
            }

            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(node.clone());

            while let Some(current) = queue.pop_front() {
                if visited.contains(&current) {
                    continue;
                }
                visited.insert(current.clone());
                component.push(current.clone());

                if let Some(neighbors) = adj.get(&current) {
                    for neighbor in neighbors {
                        if !visited.contains(neighbor) {
                            queue.push_back(neighbor.clone());
                        }
                    }
                }
            }

            if !component.is_empty() {
                communities.insert(component_id, component);
                component_id += 1;
            }

            if communities.len() >= max_records {
                break;
            }
        }

        Ok(communities)
    }

    /// Get the most connected records (highest degree).
    pub fn get_hubs(&self, limit: usize) -> rusqlite::Result<Vec<(String, usize)>> {
        // Get all unique node IDs from edges
        let mut results: Vec<(String, usize)> = Vec::new();
        {
            let conn = self.store.conn.lock().map_err(|e| {
                rusqlite::Error::InvalidParameterName(format!("Mutex poisoned: {}", e))
            })?;
            let mut stmt = conn.prepare(
                "SELECT node, COUNT(*) as degree FROM (
                    SELECT source_id as node FROM graph_edges
                    UNION ALL
                    SELECT target_id as node FROM graph_edges
                ) GROUP BY node ORDER BY degree DESC LIMIT ?1",
            )?;
            let mut rows = stmt.query([limit as i64])?;
            while let Some(row) = rows.next()? {
                let node: String = row.get(0)?;
                let degree: i64 = row.get(1)?;
                results.push((node, degree as usize));
            }
        }

        Ok(results)
    }

    /// Delete all edges connected to a record.
    pub fn remove_edges(&self, record_id: &str) -> rusqlite::Result<u64> {
        let conn = self.store.conn.lock().map_err(|e| {
            rusqlite::Error::InvalidParameterName(format!("Mutex poisoned: {}", e))
        })?;
        let deleted = conn.execute(
            "DELETE FROM graph_edges WHERE source_id = ?1 OR target_id = ?1",
            [record_id],
        )?;
        Ok(deleted as u64)
    }

    /// Clear the entire graph.
    pub fn clear(&self) -> rusqlite::Result<()> {
        let conn = self.store.conn.lock().map_err(|e| {
            rusqlite::Error::InvalidParameterName(format!("Mutex poisoned: {}", e))
        })?;
        conn.execute("DELETE FROM graph_edges", [])?;
        Ok(())
    }

    /// Count total edges in the graph.
    pub fn edge_count(&self) -> rusqlite::Result<u64> {
        let conn = self.store.conn.lock().map_err(|e| {
            rusqlite::Error::InvalidParameterName(format!("Mutex poisoned: {}", e))
        })?;
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM graph_edges", [], |r| r.get(0))?;
        Ok(count as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::StorageConfig;

    fn setup_graph() -> (MemoryStore, KnowledgeGraph) {
        let config = StorageConfig::default();
        let store = MemoryStore::open(&config).unwrap();
        let graph = KnowledgeGraph::new(store.clone());

        // Insert records
        for id in &["a", "b", "c", "d", "e"] {
            let record = MemoryRecord::new(id.to_string(), format!("Node {}", id), "graph_test".into());
            store.insert(&record).unwrap();
        }

        // Create edges: a → b, b → c, a → c, a → d
        graph.add_edge("a", "b", "related_to", 0.9).unwrap();
        graph.add_edge("b", "c", "related_to", 0.8).unwrap();
        graph.add_edge("a", "c", "depends_on", 0.7).unwrap();
        graph.add_edge("a", "d", "related_to", 0.5).unwrap();

        (store, graph)
    }

    #[test]
    fn test_edge_creation() {
        let (_, graph) = setup_graph();
        let edges = graph.get_edges("a").unwrap();
        assert_eq!(edges.len(), 3);
    }

    #[test]
    fn test_bfs_traversal() {
        let (_, graph) = setup_graph();
        let results = graph.bfs("a", 2, None).unwrap();

        // Starting from a, depth 0: a
        // Depth 1: b, c, d
        // Depth 2: c (from b)
        assert!(results.len() >= 4);
    }

    #[test]
    fn test_degree() {
        let (_, graph) = setup_graph();
        let deg = graph.degree("a").unwrap();
        assert_eq!(deg, 3); // connected to b, c, d
    }

    #[test]
    fn test_find_path() {
        let (_, graph) = setup_graph();
        let path = graph.find_path("a", "c", 3).unwrap();
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.node_id, "c");
    }

    #[test]
    fn test_communities() {
        let (_, graph) = setup_graph();
        let communities = graph.detect_communities(10).unwrap();
        assert_eq!(communities.len(), 1); // all connected in one community
    }

    #[test]
    fn test_hubs() {
        let (_, graph) = setup_graph();
        let hubs = graph.get_hubs(5).unwrap();
        assert_eq!(hubs[0].0, "a"); // a has the most connections
        assert_eq!(hubs[0].1, 3);
    }
}
