"""
Dependency injection container for the evaluation framework.
"""

import logging
from typing import Any, Dict, Type, TypeVar, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from interfaces.core_interfaces import (
    LLMInterface, EmbeddingInterface, QueryRunnerInterface,
    MetricInterface, ConfigurationInterface, DataLoaderInterface,
    ResourceManagerInterface, StatisticalAnalyzerInterface
)


T = TypeVar('T')


class ServiceLifetime(Enum):
    """Service lifetime management."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


@dataclass
class ServiceDescriptor:
    """Describes how to create and manage a service."""
    interface: Type
    implementation: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT
    dependencies: Dict[str, str] = field(default_factory=dict)


class DependencyContainer:
    """Dependency injection container."""
    
    def __init__(self):
        self._services: Dict[str, ServiceDescriptor] = {}
        self._instances: Dict[str, Any] = {}
        self._logger = logging.getLogger(__name__)
    
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> 'DependencyContainer':
        """Register a singleton service."""
        service_name = interface.__name__
        self._services[service_name] = ServiceDescriptor(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SINGLETON
        )
        return self
    
    def register_transient(self, interface: Type[T], implementation: Type[T], 
                          dependencies: Optional[Dict[str, str]] = None) -> 'DependencyContainer':
        """Register a transient service."""
        service_name = interface.__name__
        self._services[service_name] = ServiceDescriptor(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.TRANSIENT,
            dependencies=dependencies or {}
        )
        return self
    
    def register_singleton_with_deps(self, interface: Type[T], implementation: Type[T], 
                                   dependencies: Optional[Dict[str, str]] = None) -> 'DependencyContainer':
        """Register a singleton service with dependencies."""
        service_name = interface.__name__
        self._services[service_name] = ServiceDescriptor(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SINGLETON,
            dependencies=dependencies or {}
        )
        return self
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T], 
                        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT) -> 'DependencyContainer':
        """Register a service factory."""
        service_name = interface.__name__
        self._services[service_name] = ServiceDescriptor(
            interface=interface,
            factory=factory,
            lifetime=lifetime
        )
        return self
    
    def register_instance(self, interface: Type[T], instance: T) -> 'DependencyContainer':
        """Register a service instance."""
        service_name = interface.__name__
        self._services[service_name] = ServiceDescriptor(
            interface=interface,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON
        )
        self._instances[service_name] = instance
        return self
    
    def get(self, interface: Type[T]) -> T:
        """Get a service instance."""
        service_name = interface.__name__
        
        if service_name not in self._services:
            raise ValueError(f"Service {service_name} not registered")
        
        descriptor = self._services[service_name]
        
        # Return existing instance for singletons
        if descriptor.lifetime == ServiceLifetime.SINGLETON and service_name in self._instances:
            return self._instances[service_name]
        
        # Create new instance
        instance = self._create_instance(descriptor)
        
        # Store singleton instances
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            self._instances[service_name] = instance
        
        return instance
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create a service instance."""
        if descriptor.instance is not None:
            return descriptor.instance
        
        if descriptor.factory is not None:
            return descriptor.factory()
        
        if descriptor.implementation is not None:
            # Get constructor dependencies if any
            dependencies = {}
            if descriptor.dependencies:
                for param_name, service_name in descriptor.dependencies.items():
                    if service_name in self._services:
                        dependencies[param_name] = self.get_by_name(service_name)
                    else:
                        self._logger.warning(f"Dependency {service_name} not found for parameter {param_name}")
            
            # Try to create instance with dependencies
            try:
                if dependencies:
                    return descriptor.implementation(**dependencies)
                else:
                    return descriptor.implementation()
            except TypeError as e:
                self._logger.error(f"Failed to create instance of {descriptor.implementation}: {e}")
                # Fallback to parameterless constructor
                try:
                    return descriptor.implementation()
                except Exception as fallback_error:
                    raise ValueError(f"Cannot create instance for {descriptor.interface}: {fallback_error}")
        
        raise ValueError(f"Cannot create instance for {descriptor.interface}")
    
    def get_by_name(self, service_name: str) -> Any:
        """Get a service instance by name."""
        if service_name not in self._services:
            raise ValueError(f"Service {service_name} not registered")
        
        descriptor = self._services[service_name]
        
        # Return existing instance for singletons
        if descriptor.lifetime == ServiceLifetime.SINGLETON and service_name in self._instances:
            return self._instances[service_name]
        
        # Create new instance
        instance = self._create_instance(descriptor)
        
        # Store singleton instances
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            self._instances[service_name] = instance
        
        return instance
    
    def is_registered(self, interface: Type) -> bool:
        """Check if a service is registered."""
        return interface.__name__ in self._services
    
    def clear(self) -> None:
        """Clear all services and instances."""
        self._services.clear()
        self._instances.clear()


class ServiceProvider:
    """Service provider for accessing dependencies."""
    
    def __init__(self, container: DependencyContainer):
        self._container = container
    
    def get_llm(self) -> LLMInterface:
        """Get LLM service."""
        return self._container.get(LLMInterface)
    
    def get_embeddings(self) -> EmbeddingInterface:
        """Get embeddings service."""
        return self._container.get(EmbeddingInterface)
    
    def get_query_runner(self, method: str) -> QueryRunnerInterface:
        """Get query runner for specific method."""
        # This would be enhanced to return method-specific runners
        return self._container.get(QueryRunnerInterface)
    
    def get_configuration(self) -> ConfigurationInterface:
        """Get configuration service."""
        return self._container.get(ConfigurationInterface)
    
    def get_data_loader(self) -> DataLoaderInterface:
        """Get data loader service."""
        return self._container.get(DataLoaderInterface)
    
    def get_resource_manager(self) -> ResourceManagerInterface:
        """Get resource manager service."""
        return self._container.get(ResourceManagerInterface)
    
    def get_statistical_analyzer(self) -> StatisticalAnalyzerInterface:
        """Get statistical analyzer service."""
        return self._container.get(StatisticalAnalyzerInterface)


def create_default_container() -> DependencyContainer:
    """Create a container with default service registrations."""
    container = DependencyContainer()
    
    # Register default implementations here as they're created
    # This will be populated as we implement the concrete classes
    
    return container
