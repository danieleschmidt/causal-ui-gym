"""
WebSocket manager for real-time communication in Causal UI Gym.

This module provides WebSocket connections for real-time experiment updates,
live metrics streaming, and collaborative features.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Set, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import weakref
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types."""
    # Client to server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    EXPERIMENT_START = "experiment_start"
    EXPERIMENT_STOP = "experiment_stop"
    INTERVENTION_REQUEST = "intervention_request"
    AGENT_QUERY = "agent_query"
    HEARTBEAT = "heartbeat"
    
    # Server to client
    EXPERIMENT_UPDATE = "experiment_update"
    METRICS_UPDATE = "metrics_update"
    INTERVENTION_RESULT = "intervention_result"
    AGENT_RESPONSE = "agent_response"
    ERROR = "error"
    NOTIFICATION = "notification"
    HEARTBEAT_ACK = "heartbeat_ack"


@dataclass
class WebSocketMessage:
    """Structured WebSocket message."""
    type: MessageType
    data: Dict[str, Any]
    timestamp: datetime
    message_id: str
    user_id: Optional[str] = None
    experiment_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "user_id": self.user_id,
            "experiment_id": self.experiment_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebSocketMessage':
        """Create from dictionary."""
        return cls(
            type=MessageType(data["type"]),
            data=data["data"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_id=data["message_id"],
            user_id=data.get("user_id"),
            experiment_id=data.get("experiment_id")
        )


class WebSocketConnection:
    """Represents a WebSocket connection with metadata."""
    
    def __init__(self, websocket: WebSocket, connection_id: str, user_id: Optional[str] = None):
        self.websocket = websocket
        self.connection_id = connection_id
        self.user_id = user_id
        self.connected_at = datetime.now()
        self.last_heartbeat = datetime.now()
        self.subscriptions: Set[str] = set()
        self.message_count = 0
        self.last_message_at: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}
    
    async def send_message(self, message: WebSocketMessage) -> bool:
        """Send message to WebSocket client."""
        try:
            if self.websocket.client_state == WebSocketState.CONNECTED:
                await self.websocket.send_text(json.dumps(message.to_dict()))
                self.message_count += 1
                self.last_message_at = datetime.now()
                return True
        except Exception as e:
            logger.error(f"Error sending message to {self.connection_id}: {e}")
        return False
    
    async def send_error(self, error_message: str, error_code: str = "UNKNOWN_ERROR") -> bool:
        """Send error message to client."""
        error_msg = WebSocketMessage(
            type=MessageType.ERROR,
            data={
                "error_code": error_code,
                "message": error_message,
                "timestamp": datetime.now().isoformat()
            },
            timestamp=datetime.now(),
            message_id=str(uuid.uuid4()),
            user_id=self.user_id
        )
        return await self.send_message(error_msg)
    
    def is_alive(self, timeout_seconds: int = 30) -> bool:
        """Check if connection is still alive based on heartbeat."""
        if self.websocket.client_state != WebSocketState.CONNECTED:
            return False
        
        time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
        return time_since_heartbeat < timeout_seconds
    
    def add_subscription(self, topic: str) -> None:
        """Add subscription to a topic."""
        self.subscriptions.add(topic)
    
    def remove_subscription(self, topic: str) -> None:
        """Remove subscription from a topic."""
        self.subscriptions.discard(topic)
    
    def is_subscribed_to(self, topic: str) -> bool:
        """Check if connection is subscribed to a topic."""
        return topic in self.subscriptions


class TopicManager:
    """Manages topic subscriptions and message broadcasting."""
    
    def __init__(self):
        self.subscribers: Dict[str, Set[str]] = {}  # topic -> connection_ids
        self.connection_topics: Dict[str, Set[str]] = {}  # connection_id -> topics
    
    def subscribe(self, connection_id: str, topic: str) -> None:
        """Subscribe connection to a topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = set()
        self.subscribers[topic].add(connection_id)
        
        if connection_id not in self.connection_topics:
            self.connection_topics[connection_id] = set()
        self.connection_topics[connection_id].add(topic)
    
    def unsubscribe(self, connection_id: str, topic: str) -> None:
        """Unsubscribe connection from a topic."""
        if topic in self.subscribers:
            self.subscribers[topic].discard(connection_id)
            if not self.subscribers[topic]:
                del self.subscribers[topic]
        
        if connection_id in self.connection_topics:
            self.connection_topics[connection_id].discard(topic)
            if not self.connection_topics[connection_id]:
                del self.connection_topics[connection_id]
    
    def unsubscribe_all(self, connection_id: str) -> None:
        """Unsubscribe connection from all topics."""
        if connection_id in self.connection_topics:
            topics = self.connection_topics[connection_id].copy()
            for topic in topics:
                self.unsubscribe(connection_id, topic)
    
    def get_subscribers(self, topic: str) -> Set[str]:
        """Get all subscribers for a topic."""
        return self.subscribers.get(topic, set())
    
    def get_subscriptions(self, connection_id: str) -> Set[str]:
        """Get all subscriptions for a connection."""
        return self.connection_topics.get(connection_id, set())


class WebSocketManager:
    """Manages WebSocket connections and message routing."""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.topic_manager = TopicManager()
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0
        }
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_dead_connections())
    
    async def _cleanup_dead_connections(self) -> None:
        """Periodically clean up dead connections."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                dead_connections = []
                
                for connection_id, connection in self.connections.items():
                    if not connection.is_alive():
                        dead_connections.append(connection_id)
                
                for connection_id in dead_connections:
                    await self.disconnect(connection_id)
                    logger.info(f"Cleaned up dead connection: {connection_id}")
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None) -> str:
        """Accept new WebSocket connection."""
        connection_id = str(uuid.uuid4())
        
        try:
            await websocket.accept()
            
            connection = WebSocketConnection(
                websocket=websocket,
                connection_id=connection_id,
                user_id=user_id
            )
            
            self.connections[connection_id] = connection
            self.stats['total_connections'] += 1
            self.stats['active_connections'] = len(self.connections)
            
            logger.info(f"WebSocket connected: {connection_id} (user: {user_id})")
            
            # Send connection confirmation
            welcome_msg = WebSocketMessage(
                type=MessageType.NOTIFICATION,
                data={
                    "message": "Connected successfully",
                    "connection_id": connection_id,
                    "server_time": datetime.now().isoformat()
                },
                timestamp=datetime.now(),
                message_id=str(uuid.uuid4()),
                user_id=user_id
            )
            await connection.send_message(welcome_msg)
            
            return connection_id
            
        except Exception as e:
            logger.error(f"Error connecting WebSocket {connection_id}: {e}")
            raise
    
    async def disconnect(self, connection_id: str) -> None:
        """Disconnect WebSocket connection."""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            
            # Unsubscribe from all topics
            self.topic_manager.unsubscribe_all(connection_id)
            
            # Close WebSocket if still connected
            try:
                if connection.websocket.client_state == WebSocketState.CONNECTED:
                    await connection.websocket.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket {connection_id}: {e}")
            
            # Remove from connections
            del self.connections[connection_id]
            self.stats['active_connections'] = len(self.connections)
            
            logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def handle_message(self, connection_id: str, raw_message: str) -> None:
        """Handle incoming WebSocket message."""
        connection = self.connections.get(connection_id)
        if not connection:
            logger.error(f"Message from unknown connection: {connection_id}")
            return
        
        try:
            # Parse message
            message_data = json.loads(raw_message)
            message = WebSocketMessage.from_dict(message_data)
            
            self.stats['messages_received'] += 1
            connection.last_heartbeat = datetime.now()
            
            # Handle message based on type
            await self._handle_message_by_type(connection, message)
            
        except json.JSONDecodeError:
            await connection.send_error("Invalid JSON format", "INVALID_JSON")
            self.stats['errors'] += 1
        except KeyError as e:
            await connection.send_error(f"Missing required field: {e}", "MISSING_FIELD")
            self.stats['errors'] += 1
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
            await connection.send_error("Internal server error", "INTERNAL_ERROR")
            self.stats['errors'] += 1
    
    async def _handle_message_by_type(self, connection: WebSocketConnection, message: WebSocketMessage) -> None:
        """Handle message based on its type."""
        if message.type == MessageType.SUBSCRIBE:
            topic = message.data.get('topic')
            if topic:
                self.topic_manager.subscribe(connection.connection_id, topic)
                connection.add_subscription(topic)
                await connection.send_message(WebSocketMessage(
                    type=MessageType.NOTIFICATION,
                    data={"message": f"Subscribed to {topic}"},
                    timestamp=datetime.now(),
                    message_id=str(uuid.uuid4()),
                    user_id=connection.user_id
                ))
        
        elif message.type == MessageType.UNSUBSCRIBE:
            topic = message.data.get('topic')
            if topic:
                self.topic_manager.unsubscribe(connection.connection_id, topic)
                connection.remove_subscription(topic)
                await connection.send_message(WebSocketMessage(
                    type=MessageType.NOTIFICATION,
                    data={"message": f"Unsubscribed from {topic}"},
                    timestamp=datetime.now(),
                    message_id=str(uuid.uuid4()),
                    user_id=connection.user_id
                ))
        
        elif message.type == MessageType.HEARTBEAT:
            connection.last_heartbeat = datetime.now()
            await connection.send_message(WebSocketMessage(
                type=MessageType.HEARTBEAT_ACK,
                data={"server_time": datetime.now().isoformat()},
                timestamp=datetime.now(),
                message_id=str(uuid.uuid4()),
                user_id=connection.user_id
            ))
        
        # Execute registered handlers
        if message.type in self.message_handlers:
            for handler in self.message_handlers[message.type]:
                try:
                    await handler(connection, message)
                except Exception as e:
                    logger.error(f"Error in message handler for {message.type}: {e}")
    
    async def broadcast_to_topic(self, topic: str, message: WebSocketMessage) -> int:
        """Broadcast message to all subscribers of a topic."""
        subscribers = self.topic_manager.get_subscribers(topic)
        sent_count = 0
        
        for connection_id in subscribers:
            connection = self.connections.get(connection_id)
            if connection and await connection.send_message(message):
                sent_count += 1
        
        self.stats['messages_sent'] += sent_count
        return sent_count
    
    async def send_to_user(self, user_id: str, message: WebSocketMessage) -> int:
        """Send message to all connections for a specific user."""
        sent_count = 0
        
        for connection in self.connections.values():
            if connection.user_id == user_id:
                if await connection.send_message(message):
                    sent_count += 1
        
        self.stats['messages_sent'] += sent_count
        return sent_count
    
    async def send_to_connection(self, connection_id: str, message: WebSocketMessage) -> bool:
        """Send message to a specific connection."""
        connection = self.connections.get(connection_id)
        if connection:
            success = await connection.send_message(message)
            if success:
                self.stats['messages_sent'] += 1
            return success
        return False
    
    def register_message_handler(self, message_type: MessageType, handler: Callable) -> None:
        """Register a handler for a specific message type."""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics."""
        return {
            **self.stats,
            'connections_by_user': self._get_connections_by_user(),
            'active_topics': len(self.topic_manager.subscribers),
            'total_subscriptions': sum(len(subs) for subs in self.topic_manager.subscribers.values())
        }
    
    def _get_connections_by_user(self) -> Dict[str, int]:
        """Get connection count by user."""
        user_counts = {}
        for connection in self.connections.values():
            user_id = connection.user_id or "anonymous"
            user_counts[user_id] = user_counts.get(user_id, 0) + 1
        return user_counts
    
    async def shutdown(self) -> None:
        """Shutdown WebSocket manager and close all connections."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Close all connections
        for connection_id in list(self.connections.keys()):
            await self.disconnect(connection_id)
        
        logger.info("WebSocket manager shutdown complete")


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


# Convenience functions for broadcasting common events
async def broadcast_experiment_update(experiment_id: str, update_data: Dict[str, Any]) -> int:
    """Broadcast experiment update to subscribers."""
    message = WebSocketMessage(
        type=MessageType.EXPERIMENT_UPDATE,
        data=update_data,
        timestamp=datetime.now(),
        message_id=str(uuid.uuid4()),
        experiment_id=experiment_id
    )
    return await websocket_manager.broadcast_to_topic(f"experiment:{experiment_id}", message)


async def broadcast_metrics_update(experiment_id: str, metrics_data: Dict[str, Any]) -> int:
    """Broadcast metrics update to subscribers."""
    message = WebSocketMessage(
        type=MessageType.METRICS_UPDATE,
        data=metrics_data,
        timestamp=datetime.now(),
        message_id=str(uuid.uuid4()),
        experiment_id=experiment_id
    )
    return await websocket_manager.broadcast_to_topic(f"metrics:{experiment_id}", message)


async def send_intervention_result(user_id: str, intervention_data: Dict[str, Any]) -> int:
    """Send intervention result to a specific user."""
    message = WebSocketMessage(
        type=MessageType.INTERVENTION_RESULT,
        data=intervention_data,
        timestamp=datetime.now(),
        message_id=str(uuid.uuid4()),
        user_id=user_id
    )
    return await websocket_manager.send_to_user(user_id, message)