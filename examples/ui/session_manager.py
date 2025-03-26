import os
import json
import uuid
import datetime
from typing import Dict, List, Any, Optional

class Session:
    """
    Session class to store search results and analytics data
    """
    
    def __init__(self, name: str = "Unnamed Session", session_id: str = None):
        """
        Initialize a new session
        
        Args:
            name: Session name
            session_id: Optional session ID (will generate one if not provided)
        """
        self.id = session_id or str(uuid.uuid4())
        self.name = name
        self.created_at = datetime.datetime.now().isoformat()
        self.last_modified = self.created_at
        self.queries = []
        self.results = {}
        self.analytics = {}
        self.comparisons = {}
        self.visualizations = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the session to a dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "queries": self.queries,
            "results_count": len(self.results),
            "analytics_count": len(self.analytics),
            "comparisons_count": len(self.comparisons),
            "visualizations_count": len(self.visualizations)
        }
        
    def to_json(self) -> str:
        """Convert the session to a JSON string"""
        return json.dumps({
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "queries": self.queries,
            "results": self.results,
            "analytics": self.analytics,
            "comparisons": self.comparisons,
            "visualizations": self.visualizations
        }, indent=2)
        
    def add_query_results(self, query: str, results: List[Dict], metadata: Dict = None) -> str:
        """
        Add query results to the session
        
        Args:
            query: The search query
            results: List of search results
            metadata: Optional metadata about the search
            
        Returns:
            str: A result ID for referencing these results
        """
        result_id = str(uuid.uuid4())
        
        # Add query to list if not already present
        if query not in self.queries:
            self.queries.append(query)
            
        # Update timestamp
        self.last_modified = datetime.datetime.now().isoformat()
        
        # Store results
        self.results[result_id] = {
            "id": result_id,
            "query": query,
            "timestamp": self.last_modified,
            "results": results,
            "metadata": metadata or {}
        }
        
        return result_id
        
    def add_analytics(self, query: str, metrics: Dict, result_id: str = None) -> str:
        """
        Add analytics data to the session
        
        Args:
            query: The search query
            metrics: Dictionary of metric data
            result_id: Optional result ID that these metrics relate to
            
        Returns:
            str: An analytics ID for referencing these metrics
        """
        analytics_id = str(uuid.uuid4())
        
        # Update timestamp
        self.last_modified = datetime.datetime.now().isoformat()
        
        # Store analytics
        self.analytics[analytics_id] = {
            "id": analytics_id,
            "query": query,
            "timestamp": self.last_modified,
            "metrics": metrics,
            "result_id": result_id
        }
        
        return analytics_id
        
    def add_comparison(self, query: str, comparison_data: Dict) -> str:
        """
        Add algorithm comparison data to the session
        
        Args:
            query: The search query
            comparison_data: Dictionary of comparison data
            
        Returns:
            str: A comparison ID for referencing this comparison
        """
        comparison_id = str(uuid.uuid4())
        
        # Update timestamp
        self.last_modified = datetime.datetime.now().isoformat()
        
        # Store comparison
        self.comparisons[comparison_id] = {
            "id": comparison_id,
            "query": query,
            "timestamp": self.last_modified,
            "data": comparison_data
        }
        
        return comparison_id
        
    def add_visualization(self, query: str, visualization_data: Dict, result_id: str = None) -> str:
        """
        Add visualization data to the session
        
        Args:
            query: The search query or visualization name
            visualization_data: Dictionary of visualization data
            result_id: Optional result ID that this visualization relates to
            
        Returns:
            str: A visualization ID for referencing this visualization
        """
        visualization_id = str(uuid.uuid4())
        
        # Update timestamp
        self.last_modified = datetime.datetime.now().isoformat()
        
        # Store visualization
        self.visualizations[visualization_id] = {
            "id": visualization_id,
            "query": query,
            "timestamp": self.last_modified,
            "data": visualization_data,
            "result_id": result_id
        }
        
        return visualization_id
        
    def get_query_results(self, result_id: str) -> Optional[Dict]:
        """Get query results by ID"""
        return self.results.get(result_id)
        
    def get_analytics(self, analytics_id: str) -> Optional[Dict]:
        """Get analytics by ID"""
        return self.analytics.get(analytics_id)
        
    def get_comparison(self, comparison_id: str) -> Optional[Dict]:
        """Get comparison by ID"""
        return self.comparisons.get(comparison_id)
        
    def get_visualization(self, visualization_id: str) -> Optional[Dict]:
        """Get visualization by ID"""
        return self.visualizations.get(visualization_id)

class SessionManager:
    """
    Manager for creating, storing, and retrieving sessions
    """
    
    def __init__(self, storage_dir: str = None):
        """
        Initialize the session manager
        
        Args:
            storage_dir: Directory to store session files
        """
        self.sessions: Dict[str, Session] = {}
        
        # Set up storage directory
        if storage_dir:
            self.storage_dir = storage_dir
        else:
            # Default to a directory in the current working directory
            self.storage_dir = os.path.join(os.getcwd(), "sessions")
            
        # Create the directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Load existing sessions
        self._load_sessions()
        
    def _load_sessions(self):
        """Load sessions from storage directory"""
        if not os.path.exists(self.storage_dir):
            return
            
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                try:
                    filepath = os.path.join(self.storage_dir, filename)
                    with open(filepath, 'r') as f:
                        session_data = json.load(f)
                        
                    # Create session from data
                    session = Session(
                        name=session_data.get("name", "Unnamed Session"),
                        session_id=session_data.get("id")
                    )
                    session.created_at = session_data.get("created_at", session.created_at)
                    session.last_modified = session_data.get("last_modified", session.last_modified)
                    session.queries = session_data.get("queries", [])
                    session.results = session_data.get("results", {})
                    session.analytics = session_data.get("analytics", {})
                    session.comparisons = session_data.get("comparisons", {})
                    session.visualizations = session_data.get("visualizations", {})
                    
                    # Add session to manager
                    self.sessions[session.id] = session
                except Exception as e:
                    print(f"Error loading session from {filename}: {e}")
    
    def create_session(self, name: str = "Unnamed Session") -> Session:
        """
        Create a new session
        
        Args:
            name: Session name
            
        Returns:
            Session: The created session
        """
        session = Session(name=name)
        self.sessions[session.id] = session
        self._save_session(session)
        return session
        
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID
        
        Args:
            session_id: The session ID
            
        Returns:
            Session or None: The session, or None if not found
        """
        return self.sessions.get(session_id)
        
    def list_sessions(self) -> List[Dict]:
        """
        List all sessions
        
        Returns:
            List[Dict]: List of session summaries
        """
        return [session.to_dict() for session in self.sessions.values()]
        
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session
        
        Args:
            session_id: The session ID
            
        Returns:
            bool: True if the session was deleted, False otherwise
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            
            # Delete the session file
            filepath = os.path.join(self.storage_dir, f"{session_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
                
            return True
            
        return False
        
    def _save_session(self, session: Session):
        """
        Save a session to storage
        
        Args:
            session: The session to save
        """
        filepath = os.path.join(self.storage_dir, f"{session.id}.json")
        with open(filepath, 'w') as f:
            f.write(session.to_json())
            
    def export_session(self, session_id: str) -> Optional[str]:
        """
        Export a session to JSON
        
        Args:
            session_id: The session ID
            
        Returns:
            str or None: The session JSON, or None if not found
        """
        session = self.get_session(session_id)
        if session:
            return session.to_json()
        return None
        
    def import_session(self, json_data: str) -> Optional[Session]:
        """
        Import a session from JSON
        
        Args:
            json_data: The session JSON data
            
        Returns:
            Session or None: The imported session, or None if import failed
        """
        try:
            session_data = json.loads(json_data)
            
            # Check if this session already exists
            if "id" in session_data and session_data["id"] in self.sessions:
                # Update existing session
                session = self.sessions[session_data["id"]]
                session.name = session_data.get("name", session.name)
                session.queries = session_data.get("queries", session.queries)
                session.results = session_data.get("results", session.results)
                session.analytics = session_data.get("analytics", session.analytics)
                session.comparisons = session_data.get("comparisons", session.comparisons)
                session.visualizations = session_data.get("visualizations", session.visualizations)
            else:
                # Create new session from data
                session = Session(
                    name=session_data.get("name", "Imported Session"),
                    session_id=session_data.get("id")
                )
                session.created_at = session_data.get("created_at", session.created_at)
                session.last_modified = session_data.get("last_modified", session.last_modified)
                session.queries = session_data.get("queries", [])
                session.results = session_data.get("results", {})
                session.analytics = session_data.get("analytics", {})
                session.comparisons = session_data.get("comparisons", {})
                session.visualizations = session_data.get("visualizations", {})
                
                # Add session to manager
                self.sessions[session.id] = session
                
            # Save the session
            self._save_session(session)
            
            return session
        except Exception as e:
            print(f"Error importing session: {e}")
            return None 