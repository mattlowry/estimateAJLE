# Database Migration System Implementation

Let me enhance the application with a robust database migration system and other critical database enhancements.

## 1. Database Migration System with Alembic

First, I'll create the complete migration system structure:

### data/migrations/env.py
```python
"""
Alembic environment configuration file.
Handles connection to database and setup for migrations.
"""

import os
import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Add parent directory to path so we can import from our application
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, parent_dir)

# Import your application's metadata object
from data.models_base import Base
from config.app_config import AppConfig

# Load configuration
config = context.config
app_config = AppConfig()

# Set database URL in config
connection_url = app_config.get('DATABASE_URL', 'sqlite:///data/app.db')
config.set_main_option('sqlalchemy.url', connection_url)

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Store metadata for 'autogenerate' support
target_metadata = Base.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    # Handle PostgreSQL specific configuration if using PostgreSQL
    db_type = app_config.get('DATABASE_TYPE', 'sqlite')
    if db_type == 'postgresql':
        # Add connect_args for PostgreSQL specific options if needed
        connectable = engine_from_config(
            config.get_section(config.config_ini_section),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )
    else:
        connectable = engine_from_config(
            config.get_section(config.config_ini_section),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata,
            # Include further configuration if needed:
            # compare_type=True,  # Compare column types
            # compare_server_default=True,  # Compare default values
            # include_schemas=True  # Include schema-level objects
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### data/migrations/script.py.mako
```python
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade():
    ${upgrades if upgrades else "pass"}


def downgrade():
    ${downgrades if downgrades else "pass"}
```

### data/migrations/README.md
```markdown
# Database Migrations

This directory contains database migration scripts using Alembic for the Electrician Estimator application.

## Understanding Migrations

Database migrations allow you to evolve your database schema over time as your application evolves, without losing data. Each migration represents a discrete change to the database schema.

## Creating a New Migration

To create a new migration after changing your SQLAlchemy models:

```bash
python -m scripts.migrate make "description of your changes"
```

This will generate a new migration script in the `versions` directory. The script will include automatically generated changes based on the difference between your models and the current database schema.

## Reviewing a Migration

Always review the auto-generated migration script before applying it. You may need to manually edit it to:

- Add data migrations in addition to schema migrations
- Fix any issues with complex schema changes
- Add custom SQL operations that Alembic cannot generate

## Applying Migrations

To apply all pending migrations:

```bash
python -m scripts.migrate upgrade head
```

To apply migrations up to a specific version:

```bash
python -m scripts.migrate upgrade <revision_id>
```

## Downgrading

To revert the most recent migration:

```bash
python -m scripts.migrate downgrade -1
```

To downgrade to a specific version:

```bash
python -m scripts.migrate downgrade <revision_id>
```

## Migration History

To see the migration history:

```bash
python -m scripts.migrate history
```

## Best Practices

1. **Small, focused migrations**: Create migrations for specific changes rather than large, sweeping changes
2. **Test migrations**: Always test migrations in a development environment before applying to production
3. **Backup before migrating**: Always backup your production database before applying migrations
4. **Version control**: Keep migration scripts in version control alongside your application code
5. **Data migrations**: Include data migrations in your migration scripts when schema changes affect existing data

## Migration Workflow

The typical workflow for making database changes is:

1. Update SQLAlchemy models in your application code
2. Generate a migration script using `scripts.migrate make`
3. Review the generated migration script and make any necessary edits
4. Test the migration in your development environment
5. Apply the migration to production using `scripts.migrate upgrade`

## Handling Production Migrations

When deploying to production:

1. Always back up the database before running migrations
2. Consider having a rollback plan if something goes wrong
3. For large tables, consider batching data migrations to minimize downtime
4. Schedule migrations during low-traffic periods if possible

## Common Issues and Solutions

### "Can't locate revision identified by '1234abc'"
This usually means you're trying to run a migration that doesn't exist in the current environment.
Make sure all migration scripts are properly deployed.

### Autogenerate doesn't detect my model changes
- Make sure your models are imported (directly or indirectly) in the env.py script
- Check that your model classes inherit from the correct Base class
- Verify that your model changes are actually different from the database schema

### Migration fails part way through
If a migration fails part way through, the database may be left in an inconsistent state.
Use `alembic current` to see the current revision, then either:
- Fix the issue and re-run the migration
- Downgrade to the previous working revision, fix the migration script, and try again
```

### data/migrations/alembic.ini
```ini
# A generic, single database configuration.

[alembic]
# path to migration scripts
script_location = data/migrations

# template used to generate migration files
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d%%(second).2d_%%(slug)s

# timezone to use when rendering the date
# within the migration file as well as the filename.
# string value is passed to dateutil.tz.gettz()
# leave blank for localtime
# timezone =

# max length of characters to apply to the
# "slug" field
truncate_slug_length = 40

# set to 'true' to run the environment during
# the 'revision' command, regardless of autogenerate
# revision_environment = false

# set to 'true' to allow .pyc and .pyo files without
# a source .py file to be detected as revisions in the
# versions/ directory
# sourceless = false

# version location specification; this defaults
# to data/migrations/versions.  When using multiple version
# directories, initial revisions must be specified with --version-path
# version_locations = %(here)s/bar %(here)s/bat data/migrations/versions

# the output encoding used when revision files
# are written from script.py.mako
# output_encoding = utf-8

sqlalchemy.url = driver://user:pass@localhost/dbname


[post_write_hooks]
# post_write_hooks defines scripts or Python functions that are run
# on newly generated revision scripts.  See the documentation for further
# detail and examples

# format using "black" - use the console_scripts runner, against the "black" entrypoint
# hooks=black
# black.type=console_scripts
# black.entrypoint=black
# black.options=-l 79

# Logging configuration
[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = INFO
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

### scripts/migrate.py
```python
"""
Database migration script to manage database schema changes across application versions.

Usage:
    python -m scripts.migrate make "description of changes"  # Create new migration
    python -m scripts.migrate upgrade [version]             # Upgrade to specified version or 'head'
    python -m scripts.migrate downgrade [version]           # Downgrade to specified version
    python -m scripts.migrate history                       # Show migration history
    python -m scripts.migrate current                       # Show current revision
    python -m scripts.migrate stamp [version]               # Mark database as migrated to a specific version
    python -m scripts.migrate check                         # Check if database is up to date with migrations
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext

# Add parent directory to path so we can import from our application
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from utils.logger import logger
from config.app_config import AppConfig
from sqlalchemy import create_engine

def get_alembic_config():
    """Get the Alembic configuration."""
    alembic_ini = os.path.join(parent_dir, 'data', 'migrations', 'alembic.ini')
    config = Config(alembic_ini)
    return config

def create_migration(message):
    """Create a new migration script."""
    # Make sure versions directory exists
    versions_dir = os.path.join(parent_dir, 'data', 'migrations', 'versions')
    os.makedirs(versions_dir, exist_ok=True)
    
    config = get_alembic_config()
    command.revision(config, message=message, autogenerate=True)
    logger.info(f"Created new migration with message: {message}")
    print(f"Created new migration with message: {message}")

def upgrade_database(revision='head'):
    """Upgrade the database to the specified revision."""
    config = get_alembic_config()
    command.upgrade(config, revision)
    logger.info(f"Database upgraded to: {revision}")
    print(f"Database upgraded to: {revision}")

def downgrade_database(revision):
    """Downgrade the database to the specified revision."""
    config = get_alembic_config()
    command.downgrade(config, revision)
    logger.info(f"Database downgraded to: {revision}")
    print(f"Database downgraded to: {revision}")

def show_history():
    """Show the migration history."""
    config = get_alembic_config()
    command.history(config, verbose=True)

def show_current():
    """Show the current database revision."""
    config = get_alembic_config()
    command.current(config, verbose=True)

def stamp_database(revision):
    """
    Stamp the database with the given revision without running migrations.
    Useful for marking a database as being at a specific revision.
    """
    config = get_alembic_config()
    command.stamp(config, revision)
    logger.info(f"Database stamped as: {revision}")
    print(f"Database stamped as: {revision}")

def check_migrations():
    """Check if the database is up to date with migrations."""
    app_config = AppConfig()
    connection_url = app_config.get('DATABASE_URL', 'sqlite:///data/app.db')
    
    # Connect to the database
    engine = create_engine(connection_url)
    
    # Get the current revision
    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        current_rev = context.get_current_revision()
    
    # Get the head revision
    config = get_alembic_config()
    script = ScriptDirectory.from_config(config)
    head_rev = script.get_current_head()
    
    if current_rev == head_rev:
        print("Database is up to date.")
        logger.info("Database migration check: up to date")
        return True
    else:
        print(f"Database needs migration. Current: {current_rev or 'None'}, Head: {head_rev}")
        logger.warning(f"Database migration check: needs update. Current: {current_rev or 'None'}, Head: {head_rev}")
        return False

def main():
    """Main entry point for the migration script."""
    parser = argparse.ArgumentParser(description='Database migration tool')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Create new migration
    make_parser = subparsers.add_parser('make', help='Create a new migration')
    make_parser.add_argument('message', help='Migration message')

    # Upgrade database
    upgrade_parser = subparsers.add_parser('upgrade', help='Upgrade the database')
    upgrade_parser.add_argument('revision', nargs='?', default='head', help='Revision to upgrade to (default: head)')

    # Downgrade database
    downgrade_parser = subparsers.add_parser('downgrade', help='Downgrade the database')
    downgrade_parser.add_argument('revision', help='Revision to downgrade to')

    # Show history
    subparsers.add_parser('history', help='Show migration history')

    # Show current revision
    subparsers.add_parser('current', help='Show current revision')
    
    # Stamp database
    stamp_parser = subparsers.add_parser('stamp', help='Stamp database as being at a specific revision')
    stamp_parser.add_argument('revision', help='Revision to stamp database with')
    
    # Check migrations
    subparsers.add_parser('check', help='Check if database is up to date with migrations')

    args = parser.parse_args()

    # Create versions directory if it doesn't exist
    versions_dir = os.path.join(parent_dir, 'data', 'migrations', 'versions')
    os.makedirs(versions_dir, exist_ok=True)

    # Run the command
    if args.command == 'make':
        create_migration(args.message)
    elif args.command == 'upgrade':
        upgrade_database(args.revision)
    elif args.command == 'downgrade':
        downgrade_database(args.revision)
    elif args.command == 'history':
        show_history()
    elif args.command == 'current':
        show_current()
    elif args.command == 'stamp':
        stamp_database(args.revision)
    elif args.command == 'check':
        check_migrations()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
```

### Sample Migration - Initial Schema (data/migrations/versions/20240101_120000_initial.py)
```python
"""Initial database schema

Revision ID: a1b2c3d4e5f6
Revises: 
Create Date: 2024-01-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create customers table
    op.create_table('customers',
        sa.Column('customer_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('email', sa.String(), nullable=True),
        sa.Column('phone', sa.String(), nullable=False),
        sa.Column('secondary_phone', sa.String(), nullable=True),
        sa.Column('billing_address', sa.String(), nullable=False),
        sa.Column('customer_type', sa.String(), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('date_added', sa.DateTime(), nullable=False),
        sa.Column('last_updated', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('customer_id')
    )
    
    # Create property_details table
    op.create_table('property_details',
        sa.Column('property_id', sa.Integer(), nullable=False),
        sa.Column('customer_id', sa.Integer(), nullable=False),
        sa.Column('address', sa.String(), nullable=False),
        sa.Column('property_type', sa.String(), nullable=True),
        sa.Column('building_age', sa.Integer(), nullable=True),
        sa.Column('electrical_panel_info', sa.String(), nullable=True),
        sa.Column('wiring_type', sa.String(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['customer_id'], ['customers.customer_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('property_id')
    )
    
    # Create estimates table
    op.create_table('estimates',
        sa.Column('estimate_id', sa.Integer(), nullable=False),
        sa.Column('customer_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('total_amount', sa.Float(), nullable=False),
        sa.Column('labor_cost', sa.Float(), nullable=False),
        sa.Column('material_cost', sa.Float(), nullable=False),
        sa.Column('created_date', sa.DateTime(), nullable=False),
        sa.Column('last_modified', sa.DateTime(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('is_approved', sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(['customer_id'], ['customers.customer_id'], ),
        sa.PrimaryKeyConstraint('estimate_id')
    )
    
    # Create estimate_items table
    op.create_table('estimate_items',
        sa.Column('item_id', sa.Integer(), nullable=False),
        sa.Column('estimate_id', sa.Integer(), nullable=False),
        sa.Column('description', sa.String(), nullable=False),
        sa.Column('quantity', sa.Float(), nullable=False),
        sa.Column('unit_price', sa.Float(), nullable=False),
        sa.Column('labor_hours', sa.Float(), nullable=False),
        sa.Column('material_cost', sa.Float(), nullable=False),
        sa.Column('sort_order', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['estimate_id'], ['estimates.estimate_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('item_id')
    )
    
    # Indexes for performance
    op.create_index(op.f('ix_customers_name'), 'customers', ['name'], unique=False)
    op.create_index(op.f('ix_customers_email'), 'customers', ['email'], unique=False)
    op.create_index(op.f('ix_customers_phone'), 'customers', ['phone'], unique=False)
    op.create_index(op.f('ix_estimates_customer_id'), 'estimates', ['customer_id'], unique=False)
    op.create_index(op.f('ix_estimates_status'), 'estimates', ['status'], unique=False)


def downgrade():
    # Drop tables in reverse order to respect foreign key constraints
    op.drop_index(op.f('ix_estimates_status'), table_name='estimates')
    op.drop_index(op.f('ix_estimates_customer_id'), table_name='estimates')
    op.drop_index(op.f('ix_customers_phone'), table_name='customers')
    op.drop_index(op.f('ix_customers_email'), table_name='customers')
    op.drop_index(op.f('ix_customers_name'), table_name='customers')
    op.drop_table('estimate_items')
    op.drop_table('estimates')
    op.drop_table('property_details')
    op.drop_table('customers')
```

## 2. Enhanced Database Manager with PostgreSQL Support and Scalability

Here's the improved database_manager.py with PostgreSQL support, connection pooling, and sharding strategies:

```python
"""
Database management module that handles database connections, pooling, and sharding.
Provides an abstraction layer for different database backends.
"""

import os
import time
import hashlib
import threading
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import sqlalchemy
from sqlalchemy import create_engine, event, MetaData, Table, Column, Integer, String
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.sql import text
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from contextlib import contextmanager
from functools import wraps

from config.app_config import AppConfig
from utils.logger import logger
from utils.error_handling import DatabaseError

class DatabaseManager:
    """
    Database manager for handling database connections with support for
    multiple database types, connection pooling, and sharding.
    """
    
    # Connection pool settings
    POOL_SIZE = 5  # Default connection pool size
    MAX_OVERFLOW = 10  # Maximum overflow connections
    POOL_TIMEOUT = 30  # Timeout for getting connection from pool (seconds)
    POOL_RECYCLE = 3600  # Recycle connections after this many seconds
    
    # Retry settings
    MAX_RETRIES = 3  # Maximum retries for transient errors
    RETRY_BACKOFF = 0.5  # Initial backoff time between retries
    
    def __init__(self, db_type: str = "sqlite", db_path: str = "data/app.db", 
                enable_sharding: bool = False, pool_size: Optional[int] = None):
        """
        Initialize the database manager.
        
        Args:
            db_type: Database type ("sqlite" or "postgresql")
            db_path: Path to the database file (SQLite) or connection string (PostgreSQL)
            enable_sharding: Whether to enable database sharding
            pool_size: Size of the connection pool (overrides default)
        """
        self.db_type = db_type.lower()
        self.db_path = db_path
        self.enable_sharding = enable_sharding
        self.pool_size = pool_size or self.POOL_SIZE
        
        # Load app config
        self.app_config = AppConfig()
        
        # Session factory for creating database sessions
        self.session_factory = None
        
        # Dictionary to store configured engines for different shards
        self.engines = {}
        
        # Dictionary to store session factories for different shards
        self.session_factories = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Initialize database
        self._initialize_database()
        
        # Register events
        self._register_events()
        
        logger.info(f"Database Manager initialized with {db_type} database")
    
    def _initialize_database(self):
        """Initialize the database connection and session factory."""
        try:
            # Create the main engine
            main_engine = self._create_engine("main")
            self.engines["main"] = main_engine
            
            # Create session factory
            factory = sessionmaker(bind=main_engine)
            self.session_factory = scoped_session(factory)
            self.session_factories["main"] = self.session_factory
            
            # Create shards if enabled
            if self.enable_sharding:
                self._initialize_shards()
                
            # Verify connection
            self._verify_connection(main_engine)
        
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}", exc_info=True)
            raise DatabaseError(f"Failed to initialize database: {e}", 
                               context={"db_type": self.db_type, "db_path": self.db_path})
    
    def _create_engine(self, shard_key: str = "main") -> sqlalchemy.engine.Engine:
        """
        Create a SQLAlchemy engine with appropriate configuration.
        
        Args:
            shard_key: Shard key for the engine
            
        Returns:
            SQLAlchemy engine
        """
        # Determine connection URL
        if self.db_type == "sqlite":
            # For SQLite, use the provided path or construct one for shards
            if shard_key == "main":
                db_path = self.db_path
            else:
                # For shards, create separate database files
                base_path, ext = os.path.splitext(self.db_path)
                db_path = f"{base_path}_{shard_key}{ext}"
            
            connect_url = f"sqlite:///{db_path}"
            
            # Define SQLite connection args
            connect_args = {
                "check_same_thread": False,  # Allow multi-threading
                "timeout": 30  # Wait up to 30 seconds for the database lock
            }
            
            # For SQLite, we need to be more conservative with pooling
            pool_size = min(self.pool_size, 10)  # Limit pool size for SQLite
            
        elif self.db_type == "postgresql":
            # For PostgreSQL, parse the connection string or use environment variables
            if self.db_path.startswith("postgresql://"):
                connect_url = self.db_path
            else:
                # Construct from environment variables or defaults
                pg_host = self.app_config.get("PG_HOST", "localhost")
                pg_port = self.app_config.get("PG_PORT", "5432")
                pg_user = self.app_config.get("PG_USER", "postgres")
                pg_pass = self.app_config.get("PG_PASS", "")
                pg_db = self.app_config.get("PG_DB", "electrician_estimator")
                
                # For shards, use separate databases or schemas
                if shard_key != "main":
                    pg_db = f"{pg_db}_{shard_key}"
                
                connect_url = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
            
            # Define PostgreSQL connection args
            connect_args = {
                "client_encoding": "utf8",
                "connect_timeout": 10,
                "application_name": f"ElectricianEstimator_{shard_key}"
            }
            
            # Use configured pool size for PostgreSQL
            pool_size = self.pool_size
            
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
        
        # Create engine with appropriate pooling
        engine = create_engine(
            connect_url,
            poolclass=QueuePool,  # Use QueuePool for connection pooling
            pool_size=pool_size,
            max_overflow=self.MAX_OVERFLOW,
            pool_timeout=self.POOL_TIMEOUT,
            pool_recycle=self.POOL_RECYCLE,
            connect_args=connect_args
        )
        
        return engine
    
    def _initialize_shards(self):
        """Initialize database shards for horizontal scaling."""
        try:
            # Get shard configuration from app config
            shard_count = self.app_config.get("SHARD_COUNT", 0)
            
            if shard_count <= 0:
                logger.info("Sharding is enabled but no shards configured")
                return
            
            logger.info(f"Initializing {shard_count} database shards")
            
            # Create engines for each shard
            for i in range(shard_count):
                shard_key = f"shard_{i}"
                
                # Create engine for this shard
                shard_engine = self._create_engine(shard_key)
                self.engines[shard_key] = shard_engine
                
                # Create session factory for this shard
                shard_factory = sessionmaker(bind=shard_engine)
                self.session_factories[shard_key] = scoped_session(shard_factory)
                
                # Verify connection
                self._verify_connection(shard_engine)
                
                logger.info(f"Initialized database shard: {shard_key}")
        
        except Exception as e:
            logger.error(f"Failed to initialize shards: {e}", exc_info=True)
            raise DatabaseError(f"Failed to initialize shards: {e}")
    
    def _verify_connection(self, engine: sqlalchemy.engine.Engine):
        """
        Verify database connection by executing a simple query.
        
        Args:
            engine: SQLAlchemy engine to verify
        """
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            logger.error(f"Database connection verification failed: {e}")
            raise
    
    def _register_events(self):
        """Register event handlers for database connections."""
        # SQLite specific event handlers
        if self.db_type == "sqlite":
            # Enable foreign keys for SQLite
            @event.listens_for(sqlalchemy.engine.Engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
                cursor.execute("PRAGMA synchronous=NORMAL")  # Better performance with acceptable safety
                cursor.close()
        
        # PostgreSQL specific event handlers
        elif self.db_type == "postgresql":
            # Could add PostgreSQL specific event handlers here
            pass
    
    @contextmanager
    def get_session(self, shard_key: str = "main"):
        """
        Get a database session.
        
        Args:
            shard_key: Key for the shard to use
            
        Yields:
            SQLAlchemy session
        """
        session = None
        try:
            # Get the appropriate session factory
            if shard_key in self.session_factories:
                session_factory = self.session_factories[shard_key]
            else:
                logger.warning(f"Shard key {shard_key} not found, using main")
                session_factory = self.session_factory
            
            # Create a new session
            session = session_factory()
            
            # Yield the session for use
            yield session
            
            # Commit if no exceptions occurred
            session.commit()
            
        except Exception as e:
            # Rollback on exception
            if session:
                session.rollback()
            
            # Re-raise the exception
            raise
            
        finally:
            # Always close the session
            if session:
                session.close()
    
    def get_shard_key_for_id(self, id_value: Union[int, str]) -> str:
        """
        Determine the shard key for a given ID.
        Used for sharded databases to route queries to the right shard.
        
        Args:
            id_value: ID value to determine shard for
            
        Returns:
            Shard key
        """
        if not self.enable_sharding:
            return "main"
        
        # Get shard count from config
        shard_count = self.app_config.get("SHARD_COUNT", 0)
        if shard_count <= 0:
            return "main"
        
        # Convert ID to string if it's not already
        id_str = str(id_value)
        
        # Use a hash function to determine the shard
        # Simple modulo sharding based on ID hash
        hash_value = int(hashlib.md5(id_str.encode()).hexdigest(), 16)
        shard_index = hash_value % shard_count
        
        return f"shard_{shard_index}"
    
    def get_shard_key_for_customer(self, customer_id: int) -> str:
        """
        Determine the shard key for a customer ID.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Shard key
        """
        return self.get_shard_key_for_id(customer_id)
    
    def execute_on_all_shards(self, callback: Callable[[str, sqlalchemy.orm.Session], Any]) -> Dict[str, Any]:
        """
        Execute a callback function on all shards.
        
        Args:
            callback: Function to call with shard_key and session
            
        Returns:
            Dictionary mapping shard keys to callback results
        """
        results = {}
        
        # Execute on main shard
        with self.get_session("main") as session:
            results["main"] = callback("main", session)
        
        # Execute on other shards if sharding is enabled
        if self.enable_sharding:
            for shard_key in self.session_factories.keys():
                if shard_key == "main":
                    continue
                
                with self.get_session(shard_key) as session:
                    results[shard_key] = callback(shard_key, session)
        
        return results
    
    def with_retry(self, func):
        """
        Decorator for database operations to retry on transient errors.
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            backoff = self.RETRY_BACKOFF
            
            while True:
                try:
                    return func(*args, **kwargs)
                except OperationalError as e:
                    # Only retry for certain operational errors that might be transient
                    retry_error = False
                    
                    # SQLite locked or timeout error
                    if self.db_type == "sqlite" and ("database is locked" in str(e) or 
                                                  "database or disk is full" in str(e) or
                                                  "disk I/O error" in str(e)):
                        retry_error = True
                    
                    # PostgreSQL connection error
                    elif self.db_type == "postgresql" and ("connection" in str(e) or
                                                         "deadlock" in str(e) or
                                                         "could not connect" in str(e)):
                        retry_error = True
                    
                    if retry_error and retry_count < self.MAX_RETRIES:
                        retry_count += 1
                        logger.warning(f"Database error (attempt {retry_count}/{self.MAX_RETRIES}): {e}")
                        time.sleep(backoff)
                        backoff *= 2  # Exponential backoff
                    else:
                        raise
        
        return wrapper
    
    def close_all_connections(self):
        """Close all database connections."""
        with self._lock:
            for engine in self.engines.values():
                engine.dispose()
            
            logger.info("All database connections closed")
    
    def get_db_size(self, shard_key: str = "main") -> int:
        """
        Get the size of the database in bytes.
        
        Args:
            shard_key: Shard key
            
        Returns:
            Database size in bytes
        """
        if self.db_type == "sqlite":
            # For SQLite, get the file size
            if shard_key == "main":
                db_path = self.db_path
            else:
                base_path, ext = os.path.splitext(self.db_path)
                db_path = f"{base_path}_{shard_key}{ext}"
            
            try:
                return os.path.getsize(db_path)
            except (OSError, FileNotFoundError):
                return 0
        
        elif self.db_type == "postgresql":
            # For PostgreSQL, query the database size
            try:
                engine = self.engines.get(shard_key)
                if not engine:
                    return 0
                
                with engine.connect() as conn:
                    result = conn.execute(text(
                        "SELECT pg_database_size(current_database())"
                    ))
                    return result.scalar() or 0
            except Exception as e:
                logger.error(f"Failed to get PostgreSQL database size: {e}")
                return 0
        
        return 0
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database connections.
        
        Returns:
            Dictionary with connection statistics
        """
        stats = {}
        
        for shard_key, engine in self.engines.items():
            pool = engine.pool
            
            # Get pool statistics
            stats[shard_key] = {
                "size": pool.size(),
                "checkedin": pool.checkedin(),
                "checkedout": pool.checkedout(),
                "overflow": pool.overflow(),
                "db_size_bytes": self.get_db_size(shard_key)
            }
        
        return stats
    
    @staticmethod
    def compare_database_types() -> Dict[str, Dict[str, Any]]:
        """
        Compare SQLite and PostgreSQL features for documentation.
        
        Returns:
            Dictionary with comparison details
        """
        return {
            "sqlite": {
                "scalability": {
                    "max_db_size": "Limited (typically <100GB practical limit)",
                    "concurrent_connections": "Limited (1 writer, multiple readers)",
                    "clustering": "Not supported natively",
                    "replication": "Not supported natively",
                    "sharding": "Manual implementation only"
                },
                "performance": {
                    "read_performance": "Good for small-to-medium datasets",
                    "write_performance": "Limited by single-writer model",
                    "indexing": "Basic B-tree indexes",
                    "query_optimization": "Limited query planning"
                },
                "features": {
                    "foreign_keys": "Supported but must be enabled",
                    "transactions": "ACID compliant",
                    "triggers": "Supported",
                    "stored_procedures": "Limited support via runtime-loadable extensions",
                    "json_support": "Basic support",
                    "full_text_search": "Available through FTS extension"
                },
                "use_cases": [
                    "Small to medium applications (<100K users)",
                    "Desktop applications",
                    "Development and testing",
                    "Simple data storage needs",
                    "Low concurrency requirements"
                ],
                "advantages": [
                    "Zero configuration",
                    "No server process required",
                    "Self-contained database file",
                    "Cross-platform compatibility",
                    "Low memory footprint"
                ],
                "disadvantages": [
                    "Limited concurrency",
                    "Performance issues with large datasets",
                    "Limited advanced SQL features",
                    "No multi-server scaling",
                    "File locking issues in network filesystems"
                ]
            },
            "postgresql": {
                "scalability": {
                    "max_db_size": "Unlimited (multi-terabyte databases in production)",
                    "concurrent_connections": "High (thousands per server)",
                    "clustering": "Supported natively",
                    "replication": "Supported (streaming replication, logical replication)",
                    "sharding": "Supported via extensions (e.g., Citus)"
                },
                "performance": {
                    "read_performance": "Excellent with proper tuning",
                    "write_performance": "Very good with proper tuning",
                    "indexing": "Advanced indexing (B-tree, Hash, GiST, SP-GiST, GIN, BRIN)",
                    "query_optimization": "Sophisticated query planner"
                },
                "features": {
                    "foreign_keys": "Fully supported",
                    "transactions": "ACID compliant",
                    "triggers": "Advanced trigger support",
                    "stored_procedures": "Fully supported (PL/pgSQL, PL/Python, etc.)",
                    "json_support": "Excellent (JSONB datatype with indexing)",
                    "full_text_search": "Integrated and powerful"
                },
                "use_cases": [
                    "Enterprise applications",
                    "High traffic web applications",
                    "Data warehousing",
                    "Geographic information systems",
                    "Applications requiring complex data validation"
                ],
                "advantages": [
                    "Excellent standards compliance",
                    "Robust feature set",
                    "Advanced data types",
                    "Extensibility",
                    "Active community and commercial support",
                    "Excellent tooling ecosystem"
                ],
                "disadvantages": [
                    "Requires server setup and maintenance",
                    "Higher resource consumption than SQLite",
                    "More complex configuration",
                    "Steeper learning curve for administration"
                ]
            },
            "recommendation": {
                "small_scale": "SQLite is recommended for small installations (<100 customers, single user)",
                "medium_scale": "Consider PostgreSQL for medium installations (100-1000 customers, multiple users)",
                "large_scale": "PostgreSQL is strongly recommended for large installations (>1000 customers, multiple users)",
                "migration_path": "The application supports SQLite for development and small deployments with a migration path to PostgreSQL for production and larger deployments"
            }
        }
```

## 3. Data Validation System

Now let's implement a robust data validation system that integrates with the database layer and core domain models:

```python
"""
Data validation module for ensuring data integrity across the application.
Provides validation utilities for repositories and domain models.
"""

from typing import Dict, Any, List, Optional, Union, Type, Callable, Set, Tuple
import re
import json
from enum import Enum
from datetime import datetime
from dataclasses import is_dataclass, asdict
from uuid import UUID
from decimal import Decimal

from pydantic import BaseModel, ValidationError, validator
from pydantic.error_wrappers import ErrorWrapper
from sqlalchemy.orm import Session

from utils.logger import logger
from utils.error_handling import ContextualError

class ValidationError(ContextualError):
    """Error raised when data validation fails."""
    pass

class ValidationType(Enum):
    """Types of validation that can be performed."""
    REQUIRED = "required"
    STRING_LENGTH = "string_length"
    NUMERIC_RANGE = "numeric_range"
    PATTERN = "pattern"
    EMAIL = "email"
    PHONE = "phone"
    ENUM_VALUE = "enum_value"
    DATE_RANGE = "date_range"
    CUSTOM = "custom"

class ValidationRule:
    """Validation rule for a field."""
    
    def __init__(self, 
                field_name: str, 
                validation_type: ValidationType, 
                error_message: str = None,
                **kwargs):
        """
        Initialize a validation rule.
        
        Args:
            field_name: Name of the field to validate
            validation_type: Type of validation to perform
            error_message: Custom error message
            **kwargs: Additional parameters for the validation
        """
        self.field_name = field_name
        self.validation_type = validation_type
        self.error_message = error_message
        self.params = kwargs
    
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate data against this rule.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if field exists in data
        if self.field_name not in data:
            # If field is required, return error
            if self.validation_type == ValidationType.REQUIRED:
                return False, self.error_message or f"Field '{self.field_name}' is required"
            
            # Otherwise, skip validation
            return True, None
        
        # Get field value
        value = data[self.field_name]
        
        # Skip validation if value is None and not required
        if value is None and self.validation_type != ValidationType.REQUIRED:
            return True, None
        
        # Validate based on validation type
        if self.validation_type == ValidationType.REQUIRED:
            # Check if value is None or empty string
            is_valid = value is not None and (not isinstance(value, str) or value.strip() != "")
            error_msg = self.error_message or f"Field '{self.field_name}' is required"
            
        elif self.validation_type == ValidationType.STRING_LENGTH:
            # Check string length
            min_length = self.params.get("min_length", 0)
            max_length = self.params.get("max_length")
            
            # Convert to string if not already
            if not isinstance(value, str):
                value = str(value)
            
            is_valid = len(value) >= min_length
            if max_length is not None:
                is_valid = is_valid and len(value) <= max_length
            
            error_msg = self.error_message
            if not error_msg:
                if max_length is not None:
                    error_msg = f"Field '{self.field_name}' must be between {min_length} and {max_length} characters"
                else:
                    error_msg = f"Field '{self.field_name}' must be at least {min_length} characters"
            
        elif self.validation_type == ValidationType.NUMERIC_RANGE:
            # Check numeric range
            min_value = self.params.get("min_value")
            max_value = self.params.get("max_value")
            
            # Convert to number if string
            if isinstance(value, str) and value.strip():
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    return False, self.error_message or f"Field '{self.field_name}' must be a number"
            
            # Check if value is a number
            if not isinstance(value, (int, float, Decimal)):
                return False, self.error_message or f"Field '{self.field_name}' must be a number"
            
            # Check range
            is_valid = True
            if min_value is not None:
                is_valid = is_valid and value >= min_value
            if max_value is not None:
                is_valid = is_valid and value <= max_value
            
            error_msg = self.error_message
            if not error_msg:
                if min_value is not None and max_value is not None:
                    error_msg = f"Field '{self.field_name}' must be between {min_value} and {max_value}"
                elif min_value is not None:
                    error_msg = f"Field '{self.field_name}' must be at least {min_value}"
                elif max_value is not None:
                    error_msg = f"Field '{self.field_name}' must be at most {max_value}"
                else:
                    error_msg = f"Field '{self.field_name}' must be a valid number"
            
        elif self.validation_type == ValidationType.PATTERN:
            # Check pattern match
            pattern = self.params.get("pattern")
            if not pattern:
                return False, "No pattern specified for validation"
            
            # Convert to string if not already
            if not isinstance(value, str):
                value = str(value)
            
            # Check if the value matches the pattern
            is_valid = bool(re.match(pattern, value))
            error_msg = self.error_message or f"Field '{self.field_name}' must match the required format"
            
        elif self.validation_type == ValidationType.EMAIL:
            # Check email format
            if not isinstance(value, str):
                value = str(value)
            
            # Simple email validation pattern
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            is_valid = bool(re.match(email_pattern, value))
            error_msg = self.error_message or f"Field '{self.field_name}' must be a valid email address"
            
        elif self.validation_type == ValidationType.PHONE:
            # Check phone format
            if not isinstance(value, str):
                value = str(value)
            
            # Strip non-numeric characters
            cleaned_phone = re.sub(r'[^0-9]', '', value)
            
            # Phone validation pattern - allow different formats
            phone_pattern = self.params.get("pattern", r"^\d{10,15}$")
            is_valid = bool(re.match(phone_pattern, cleaned_phone))
            error_msg = self.error_message or f"Field '{self.field_name}' must be a valid phone number"
            
        elif self.validation_type == ValidationType.ENUM_VALUE:
            # Check if value is in allowed values
            allowed_values = self.params.get("values", [])
            is_valid = value in allowed_values
            error_msg = self.error_message or f"Field '{self.field_name}' must be one of: {', '.join(map(str, allowed_values))}"
            
        elif self.validation_type == ValidationType.DATE_RANGE:
            # Check date range
            min_date = self.params.get("min_date")
            max_date = self.params.get("max_date")
            
            # Convert string to date if needed
            if isinstance(value, str):
                try:
                    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                except ValueError:
                    return False, self.error_message or f"Field '{self.field_name}' must be a valid date"
            
            # Check if value is a date
            if not isinstance(value, datetime):
                return False, self.error_message or f"Field '{self.field_name}' must be a valid date"
            
            # Check range
            is_valid = True
            if min_date is not None:
                is_valid = is_valid and value >= min_date
            if max_date is not None:
                is_valid = is_valid and value <= max_date
            
            error_msg = self.error_message
            if not error_msg:
                if min_date is not None and max_date is not None:
                    error_msg = f"Field '{self.field_name}' must be between {min_date} and {max_date}"
                elif min_date is not None:
                    error_msg = f"Field '{self.field_name}' must be on or after {min_date}"
                elif max_date is not None:
                    error_msg = f"Field '{self.field_name}' must be on or before {max_date}"
                else:
                    error_msg = f"Field '{self.field_name}' must be a valid date"
            
        elif self.validation_type == ValidationType.CUSTOM:
            # Custom validation function
            validator_func = self.params.get("validator")
            if not validator_func or not callable(validator_func):
                return False, "No validator function specified for custom validation"
            
            # Call the validator function
            is_valid, custom_error = validator_func(value)
            error_msg = self.error_message or custom_error or f"Field '{self.field_name}' failed validation"
            
        else:
            return False, f"Unknown validation type: {self.validation_type}"
        
        return is_valid, error_msg if not is_valid else None

class DataValidator:
    """
    Validator for data structures.
    Provides methods for validating dictionaries, models, and entity objects.
    """
    
    @staticmethod
    def validate_dict(data: Dict[str, Any], rules: List[ValidationRule]) -> List[str]:
        """
        Validate a dictionary against a list of rules.
        
        Args:
            data: Dictionary to validate
            rules: List of validation rules
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        for rule in rules:
            is_valid, error = rule.validate(data)
            if not is_valid and error:
                errors.append(error)
        
        return errors
    
    @staticmethod
    def validate_pydantic_model(model_class: Type[BaseModel], data: Dict[str, Any]) -> Tuple[Optional[BaseModel], List[str]]:
        """
        Validate data against a Pydantic model.
        
        Args:
            model_class: Pydantic model class
            data: Data to validate
            
        Returns:
            Tuple of (validated_model, error_messages)
        """
        try:
            # Validate data using Pydantic
            model = model_class(**data)
            return model, []
        except ValidationError as e:
            # Extract error messages
            error_messages = []
            for error in e.errors():
                field = ".".join(error["loc"])
                message = error["msg"]
                error_messages.append(f"{field}: {message}")
            
            return None, error_messages
    
    @staticmethod
    def validate_domain_model(model: Any, rules: List[ValidationRule]) -> List[str]:
        """
        Validate a domain model against a list of rules.
        
        Args:
            model: Domain model to validate
            rules: List of validation rules
            
        Returns:
            List of validation error messages (empty if valid)
        """
        # Convert model to dictionary
        if is_dataclass(model):
            data = asdict(model)
        elif hasattr(model, "dict") and callable(model.dict):
            data = model.dict()
        elif hasattr(model, "__dict__"):
            data = model.__dict__
        else:
            # Try to convert object to dictionary by getting attributes
            data = {}
            for attr in dir(model):
                if not attr.startswith("_") and not callable(getattr(model, attr)):
                    data[attr] = getattr(model, attr)
        
        # Validate the dictionary
        return DataValidator.validate_dict(data, rules)

class RepositoryValidator:
    """
    Validator for repository operations.
    Provides validation methods for common repository operations.
    """
    
    def __init__(self, session: Session, model_class: Any = None, validation_rules: Dict[str, List[ValidationRule]] = None):
        """
        Initialize a repository validator.
        
        Args:
            session: SQLAlchemy session
            model_class: SQLAlchemy model class
            validation_rules: Dictionary mapping operation names to validation rules
        """
        self.session = session
        self.model_class = model_class
        self.validation_rules = validation_rules or {}
    
    def validate_create(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate data for a create operation.
        
        Args:
            data: Data to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        rules = self.validation_rules.get("create", [])
        return DataValidator.validate_dict(data, rules)
    
    def validate_update(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate data for an update operation.
        
        Args:
            data: Data to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        rules = self.validation_rules.get("update", [])
        return DataValidator.validate_dict(data, rules)
    
    def validate_unique_constraint(self, field: str, value: Any, exclude_id: Optional[Any] = None) -> bool:
        """
        Validate that a field value is unique in the database.
        
        Args:
            field: Field name to check
            value: Value to check
            exclude_id: ID to exclude from the check (for updates)
            
        Returns:
            True if the value is unique, False otherwise
        """
        if not self.model_class:
            raise ValueError("Model class is required for unique constraint validation")
        
        # Create query to check uniqueness
        query = self.session.query(self.model_class).filter(getattr(self.model_class, field) == value)
        
        # Exclude current record if updating
        if exclude_id is not None:
            primary_key = self.model_class.__mapper__.primary_key[0].name
            query = query.filter(getattr(self.model_class, primary_key) != exclude_id)
        
        # Check if any records exist
        return query.first() is None

# Common validation rules for reuse
class CommonValidators:
    """Common validation rules that can be reused across the application."""
    
    @staticmethod
    def email_validator() -> ValidationRule:
        """Email validation rule."""
        return ValidationRule(
            field_name="email",
            validation_type=ValidationType.EMAIL,
            error_message="Email address must be in a valid format"
        )
    
    @staticmethod
    def phone_validator() -> ValidationRule:
        """Phone validation rule."""
        return ValidationRule(
            field_name="phone",
            validation_type=ValidationType.PHONE,
            error_message="Phone number must be in a valid format",
            pattern=r"^\d{10,15}$"  # 10-15 digits
        )
    
    @staticmethod
    def name_validator(field_name="name", min_length=2, max_length=100) -> ValidationRule:
        """Name validation rule."""
        return ValidationRule(
            field_name=field_name,
            validation_type=ValidationType.STRING_LENGTH,
            error_message=f"{field_name.capitalize()} must be between {min_length} and {max_length} characters",
            min_length=min_length,
            max_length=max_length
        )
    
    @staticmethod
    def price_validator(field_name="price", min_value=0, max_value=1000000) -> ValidationRule:
        """Price validation rule."""
        return ValidationRule(
            field_name=field_name,
            validation_type=ValidationType.NUMERIC_RANGE,
            error_message=f"{field_name.capitalize()} must be between {min_value} and {max_value}",
            min_value=min_value,
            max_value=max_value
        )
    
    @staticmethod
    def date_validator(field_name="date") -> ValidationRule:
        """Date validation rule."""
        # Custom function to check if value is a valid date
        def is_valid_date(value):
            if isinstance(value, datetime):
                return True, None
            
            if isinstance(value, str):
                try:
                    datetime.fromisoformat(value.replace('Z', '+00:00'))
                    return True, None
                except ValueError:
                    return False, f"{field_name.capitalize()} must be a valid date"
            
            return False, f"{field_name.capitalize()} must be a valid date"
        
        return ValidationRule(
            field_name=field_name,
            validation_type=ValidationType.CUSTOM,
            error_message=f"{field_name.capitalize()} must be a valid date",
            validator=is_valid_date
        )

# Validation rule sets for specific entities
def customer_validation_rules() -> Dict[str, List[ValidationRule]]:
    """Get validation rules for customers."""
    return {
        "create": [
            ValidationRule("name", ValidationType.REQUIRED),
            ValidationRule("name", ValidationType.STRING_LENGTH, min_length=2, max_length=100),
            ValidationRule("email", ValidationType.EMAIL),
            ValidationRule("phone", ValidationType.PHONE),
            ValidationRule("customer_type", ValidationType.ENUM_VALUE, values=["Residential", "Commercial", "Industrial"])
        ],
        "update": [
            ValidationRule("name", ValidationType.STRING_LENGTH, min_length=2, max_length=100),
            ValidationRule("email", ValidationType.EMAIL),
            ValidationRule("phone", ValidationType.PHONE),
            ValidationRule("customer_type", ValidationType.ENUM_VALUE, values=["Residential", "Commercial", "Industrial"])
        ]
    }

def estimate_validation_rules() -> Dict[str, List[ValidationRule]]:
    """Get validation rules for estimates."""
    return {
        "create": [
            ValidationRule("customer_id", ValidationType.REQUIRED),
            ValidationRule("title", ValidationType.REQUIRED),
            ValidationRule("title", ValidationType.STRING_LENGTH, min_length=2, max_length=200),
            ValidationRule("total_amount", ValidationType.NUMERIC_RANGE, min_value=0),
            ValidationRule("labor_cost", ValidationType.NUMERIC_RANGE, min_value=0),
            ValidationRule("material_cost", ValidationType.NUMERIC_RANGE, min_value=0),
            ValidationRule("status", ValidationType.ENUM_VALUE, values=["draft", "pending", "approved", "rejected", "completed"])
        ],
        "update": [
            ValidationRule("title", ValidationType.STRING_LENGTH, min_length=2, max_length=200),
            ValidationRule("total_amount", ValidationType.NUMERIC_RANGE, min_value=0),
            ValidationRule("labor_cost", ValidationType.NUMERIC_RANGE, min_value=0),
            ValidationRule("material_cost", ValidationType.NUMERIC_RANGE, min_value=0),
            ValidationRule("status", ValidationType.ENUM_VALUE, values=["draft", "pending", "approved", "rejected", "completed"])
        ]
    }

def estimate_line_item_validation_rules() -> Dict[str, List[ValidationRule]]:
    """Get validation rules for estimate line items."""
    return {
        "create": [
            ValidationRule("estimate_id", ValidationType.REQUIRED),
            ValidationRule("description", ValidationType.REQUIRED),
            ValidationRule("description", ValidationType.STRING_LENGTH, min_length=2, max_length=500),
            ValidationRule("quantity", ValidationType.NUMERIC_RANGE, min_value=0),
            ValidationRule("unit_price", ValidationType.NUMERIC_RANGE, min_value=0),
            ValidationRule("labor_hours", ValidationType.NUMERIC_RANGE, min_value=0)
        ],
        "update": [
            ValidationRule("description", ValidationType.STRING_LENGTH, min_length=2, max_length=500),
            ValidationRule("quantity", ValidationType.NUMERIC_RANGE, min_value=0),
            ValidationRule("unit_price", ValidationType.NUMERIC_RANGE, min_value=0),
            ValidationRule("labor_hours", ValidationType.NUMERIC_RANGE, min_value=0)
        ]
    }
```

## 4. Caching Layer Implementation

Now, let's implement a robust caching layer for the application:

```python
"""
Caching module for improving application performance by caching frequently accessed data.
Supports both in-memory and Redis caching with flexible configuration.
"""

import json
import pickle
import time
import threading
import hashlib
from typing import Dict, Any, Optional, Callable, List, Tuple, Union, Set
from enum import Enum
from datetime import datetime, timedelta
from functools import wraps
import logging

from utils.logger import logger
from config.app_config import AppConfig

# Optional Redis import
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. In-memory cache will be used instead.")

class CacheType(Enum):
    """Types of cache storage."""
    MEMORY = "memory"
    REDIS = "redis"

class CacheKeyPrefix(Enum):
    """Prefixes for cache keys to organize the cache."""
    CUSTOMER = "customer"
    ESTIMATE = "estimate"
    MATERIAL = "material"
    IMAGE = "image"
    AI_RESPONSE = "ai"
    SETTINGS = "settings"
    OTHER = "other"

class CachedObject:
    """Object stored in the cache with metadata."""
    
    def __init__(self, value: Any, ttl: Optional[int] = None):
        """
        Initialize a cached object.
        
        Args:
            value: The value to cache
            ttl: Time-to-live in seconds, or None for no expiration
        """
        self.value = value
        self.created_at = time.time()
        self.expires_at = time.time() + ttl if ttl is not None else None
        self.access_count = 0
        self.last_accessed_at = self.created_at
    
    def is_expired(self) -> bool:
        """
        Check if the cached object is expired.
        
        Returns:
            True if expired, False otherwise
        """
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self):
        """Update the access metadata for the cached object."""
        self.access_count += 1
        self.last_accessed_at = time.time()

class CacheStats:
    """Statistics for the cache."""
    
    def __init__(self):
        """Initialize cache statistics."""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.clear_count = 0
        self.last_reset = time.time()
    
    def reset(self):
        """Reset cache statistics."""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.clear_count = 0
        self.last_reset = time.time()
    
    def hit_rate(self) -> float:
        """
        Calculate the cache hit rate.
        
        Returns:
            Hit rate as a float between 0 and 1
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert statistics to a dictionary.
        
        Returns:
            Dictionary of cache statistics
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "clear_count": self.clear_count,
            "hit_rate": self.hit_rate(),
            "last_reset": datetime.fromtimestamp(self.last_reset).isoformat()
        }

class MemoryCache:
    """In-memory cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the memory cache.
        
        Args:
            max_size: Maximum number of items to store
        """
        self.cache: Dict[str, CachedObject] = {}
        self.max_size = max_size
        self.lock = threading.RLock()
        self.stats = CacheStats()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        with self.lock:
            # Check if key exists
            if key not in self.cache:
                self.stats.misses += 1
                return None
            
            # Get cached object
            cached = self.cache[key]
            
            # Check if expired
            if cached.is_expired():
                self.stats.misses += 1
                del self.cache[key]
                return None
            
            # Update access metadata
            cached.touch()
            self.stats.hits += 1
            
            return cached.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds, or None for no expiration
        """
        with self.lock:
            # Check if we need to evict items
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_one()
            
            # Cache the object
            self.cache[key] = CachedObject(value, ttl)
            self.stats.sets += 1
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was deleted, False otherwise
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats.deletes += 1
                return True
            return False
    
    def clear(self):
        """Clear the entire cache."""
        with self.lock:
            self.cache.clear()
            self.stats.clear_count += 1
    
    def clear_with_prefix(self, prefix: str):
        """
        Clear all keys with the given prefix.
        
        Args:
            prefix: Key prefix to clear
        """
        with self.lock:
            keys_to_delete = [k for k in self.cache.keys() if k.startswith(prefix)]
            for key in keys_to_delete:
                del self.cache[key]
            
            if keys_to_delete:
                self.stats.deletes += len(keys_to_delete)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        with self.lock:
            stats = self.stats.to_dict()
            stats["size"] = len(self.cache)
            stats["max_size"] = self.max_size
            
            # Calculate memory usage (approximate)
            memory_usage = 0
            for k, v in self.cache.items():
                # Approximate size of key
                memory_usage += len(k) * 2  # Unicode string size
                
                # Approximate size of value (very rough)
                try:
                    memory_usage += len(pickle.dumps(v.value))
                except:
                    # If can't pickle, use a conservative estimate
                    memory_usage += 1024  # 1KB per item as fallback
            
            stats["memory_usage_bytes"] = memory_usage
            
            return stats
    
    def _evict_one(self):
        """Evict one item from the cache based on policy."""
        # Simple LRU policy - evict least recently used item
        oldest_key = None
        oldest_time = float('inf')
        
        for key, obj in self.cache.items():
            if obj.last_accessed_at < oldest_time:
                oldest_time = obj.last_accessed_at
                oldest_key = key
        
        if oldest_key:
            del self.cache[oldest_key]

class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self, 
                host: str = "localhost", 
                port: int = 6379, 
                db: int = 0, 
                password: Optional[str] = None,
                prefix: str = "cache:"):
        """
        Initialize the Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database
            password: Redis password
            prefix: Key prefix for all cache keys
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Please install the redis package.")
        
        self.redis = redis.Redis(
            host=host, 
            port=port, 
            db=db, 
            password=password,
            decode_responses=False  # Keep binary data
        )
        self.prefix = prefix
        self.stats = CacheStats()
        
        # Try to connect to Redis
        try:
            self.redis.ping()
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            raise
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        redis_key = self._make_key(key)
        
        try:
            # Get value from Redis
            value = self.redis.get(redis_key)
            
            # If not found, return None
            if value is None:
                self.stats.misses += 1
                return None
            
            # Deserialize value
            result = pickle.loads(value)
            
            # Update stats
            self.stats.hits += 1
            
            # Update the access count in a separate Redis call
            self.redis.hincrby(f"{self.prefix}stats", f"hits:{key}", 1)
            
            return result
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.stats.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds, or None for no expiration
        """
        redis_key = self._make_key(key)
        
        try:
            # Serialize value
            serialized = pickle.dumps(value)
            
            # Set in Redis
            if ttl is not None:
                self.redis.setex(redis_key, ttl, serialized)
            else:
                self.redis.set(redis_key, serialized)
            
            # Update stats
            self.stats.sets += 1
            
            # Record creation time
            self.redis.hset(f"{self.prefix}metadata", key, json.dumps({
                "created_at": time.time(),
                "ttl": ttl
            }))
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was deleted, False otherwise
        """
        redis_key = self._make_key(key)
        
        try:
            # Delete from Redis
            result = self.redis.delete(redis_key) > 0
            
            if result:
                self.stats.deletes += 1
                
                # Delete metadata
                self.redis.hdel(f"{self.prefix}metadata", key)
            
            return result
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear(self):
        """Clear all cache entries with this prefix."""
        try:
            # Find all keys with this prefix
            cursor = 0
            pattern = f"{self.prefix}*"
            total_deleted = 0
            
            while True:
                cursor, keys = self.redis.scan(cursor, match=pattern)
                if keys:
                    total_deleted += self.redis.delete(*keys)
                
                if cursor == 0:
                    break
            
            self.stats.clear_count += 1
            self.stats.deletes += total_deleted
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def clear_with_prefix(self, prefix: str):
        """
        Clear all keys with the given prefix.
        
        Args:
            prefix: Key prefix to clear
        """
        redis_prefix = f"{self.prefix}{prefix}"
        
        try:
            # Find all keys with this prefix
            cursor = 0
            pattern = f"{redis_prefix}*"
            total_deleted = 0
            
            while True:
                cursor, keys = self.redis.scan(cursor, match=pattern)
                if keys:
                    total_deleted += self.redis.delete(*keys)
                
                if cursor == 0:
                    break
            
            self.stats.deletes += total_deleted
        except Exception as e:
            logger.error(f"Redis clear with prefix error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        stats = self.stats.to_dict()
        
        try:
            # Get Redis info
            info = self.redis.info()
            stats["redis_memory_used_bytes"] = info.get("used_memory", 0)
            stats["redis_connected_clients"] = info.get("connected_clients", 0)
            stats["redis_uptime_seconds"] = info.get("uptime_in_seconds", 0)
            
            # Count keys with our prefix
            cursor = 0
            pattern = f"{self.prefix}*"
            key_count = 0
            
            while True:
                cursor, keys = self.redis.scan(cursor, match=pattern, count=1000)
                key_count += len(keys)
                
                if cursor == 0:
                    break
            
            stats["size"] = key_count
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
        
        return stats
    
    def _make_key(self, key: str) -> str:
        """
        Create a Redis key with the prefix.
        
        Args:
            key: Original key
            
        Returns:
            Key with prefix
        """
        return f"{self.prefix}{key}"

class CacheManager:
    """
    Cache manager for the application.
    Provides a unified interface for caching with configurable backend.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the cache manager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the cache manager."""
        if CacheManager._instance is not None:
            raise RuntimeError("Use CacheManager.get_instance() to get the singleton instance")
        
        # Load configuration
        self.app_config = AppConfig()
        self.cache_type = CacheType(self.app_config.get("CACHE_TYPE", "memory").lower())
        
        # Set up the cache backend
        if self.cache_type == CacheType.REDIS and REDIS_AVAILABLE:
            try:
                self.cache = RedisCache(
                    host=self.app_config.get("REDIS_HOST", "localhost"),
                    port=int(self.app_config.get("REDIS_PORT", 6379)),
                    db=int(self.app_config.get("REDIS_DB", 0)),
                    password=self.app_config.get("REDIS_PASSWORD"),
                    prefix=self.app_config.get("CACHE_PREFIX", "estimator:")
                )
                logger.info("Using Redis cache backend")
            except Exception as e:
                logger.error(f"Failed to initialize Redis cache: {e}")
                logger.info("Falling back to in-memory cache")
                self.cache_type = CacheType.MEMORY
                self.cache = MemoryCache(max_size=int(self.app_config.get("MEMORY_CACHE_MAX_SIZE", 10000)))
        else:
            self.cache_type = CacheType.MEMORY
            self.cache = MemoryCache(max_size=int(self.app_config.get("MEMORY_CACHE_MAX_SIZE", 10000)))
            logger.info("Using in-memory cache backend")
        
        # Set default TTL from config
        self.default_ttl = int(self.app_config.get("CACHE_DEFAULT_TTL", 3600))  # 1 hour
        
        # Configure entity-specific TTLs
        self.ttls = {
            CacheKeyPrefix.CUSTOMER.value: int(self.app_config.get("CACHE_CUSTOMER_TTL", self.default_ttl)),
            CacheKeyPrefix.ESTIMATE.value: int(self.app_config.get("CACHE_ESTIMATE_TTL", self.default_ttl)),
            CacheKeyPrefix.MATERIAL.value: int(self.app_config.get("CACHE_MATERIAL_TTL", self.default_ttl * 24)),  # 24h
            CacheKeyPrefix.IMAGE.value: int(self.app_config.get("CACHE_IMAGE_TTL", self.default_ttl * 24)),  # 24h
            CacheKeyPrefix.AI_RESPONSE.value: int(self.app_config.get("CACHE_AI_TTL", self.default_ttl * 24 * 7)),  # 7d
            CacheKeyPrefix.SETTINGS.value: int(self.app_config.get("CACHE_SETTINGS_TTL", self.default_ttl * 24)),  # 24h
            CacheKeyPrefix.OTHER.value: self.default_ttl
        }
        
        # Keep track of cache keys by entity type
        self.entity_keys: Dict[str, Set[str]] = {prefix.value: set() for prefix in CacheKeyPrefix}
        
        logger.info(f"Cache Manager initialized with {self.cache_type.value} backend")
    
    def make_key(self, prefix: Union[str, CacheKeyPrefix], *parts: str) -> str:
        """
        Create a cache key.
        
        Args:
            prefix: Key prefix (entity type)
            *parts: Key parts
            
        Returns:
            Cache key
        """
        # Handle prefix as enum
        if isinstance(prefix, CacheKeyPrefix):
            prefix = prefix.value
            
        # Join parts with colons
        key_parts = [prefix] + [str(part) for part in parts if part]
        return ":".join(key_parts)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        value = self.cache.get(key)
        return value if value is not None else default
    
    def get_entity(self, prefix: Union[str, CacheKeyPrefix], entity_id: Union[int, str], default: Any = None) -> Any:
        """
        Get an entity from the cache.
        
        Args:
            prefix: Entity type prefix
            entity_id: Entity ID
            default: Default value if not found
            
        Returns:
            Cached entity or default
        """
        key = self.make_key(prefix, str(entity_id))
        return self.get(key, default)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds, or None to use the default TTL
        """
        # Get the appropriate TTL
        if ttl is None:
            # Try to determine TTL from key prefix
            parts = key.split(":", 1)
            prefix = parts[0] if parts else ""
            ttl = self.ttls.get(prefix, self.default_ttl)
        
        # Cache the value
        self.cache.set(key, value, ttl)
        
        # Record the key by entity type
        for prefix in self.entity_keys:
            if key.startswith(f"{prefix}:"):
                self.entity_keys[prefix].add(key)
                break
    
    def set_entity(self, prefix: Union[str, CacheKeyPrefix], entity_id: Union[int, str], value: Any, ttl: Optional[int] = None):
        """
        Set an entity in the cache.
        
        Args:
            prefix: Entity type prefix
            entity_id: Entity ID
            value: Value to cache
            ttl: Time-to-live in seconds, or None to use the default TTL
        """
        key = self.make_key(prefix, str(entity_id))
        self.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was deleted, False otherwise
        """
        result = self.cache.delete(key)
        
        # Remove from entity keys tracking
        for prefix in self.entity_keys:
            if key in self.entity_keys[prefix]:
                self.entity_keys[prefix].remove(key)
        
        return result
    
    def delete_entity(self, prefix: Union[str, CacheKeyPrefix], entity_id: Union[int, str]) -> bool:
        """
        Delete an entity from the cache.
        
        Args:
            prefix: Entity type prefix
            entity_id: Entity ID
            
        Returns:
            True if the entity was deleted, False otherwise
        """
        key = self.make_key(prefix, str(entity_id))
        return self.delete(key)
    
    def clear(self):
        """Clear the entire cache."""
        self.cache.clear()
        
        # Reset entity keys tracking
        for prefix in self.entity_keys:
            self.entity_keys[prefix].clear()
    
    def clear_entity_type(self, prefix: Union[str, CacheKeyPrefix]):
        """
        Clear all entities of a given type.
        
        Args:
            prefix: Entity type prefix
        """
        # Handle prefix as enum
        if isinstance(prefix, CacheKeyPrefix):
            prefix = prefix.value
            
        self.cache.clear_with_prefix(f"{prefix}:")
        
        # Reset entity keys tracking for this prefix
        if prefix in self.entity_keys:
            self.entity_keys[prefix].clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        stats = self.cache.get_stats()
        
        # Add entity counts
        entity_counts = {prefix: len(keys) for prefix, keys in self.entity_keys.items()}
        stats["entity_counts"] = entity_counts
        
        return stats
    
    def touch(self, key: str):
        """
        Update the access time for a key without fetching it.
        Only used internally for specific cache implementations.
        
        Args:
            key: Cache key
        """
        if self.cache_type == CacheType.MEMORY:
            with self.cache.lock:
                if key in self.cache.cache:
                    self.cache.cache[key].touch()
    
    def refresh_entity(self, prefix: Union[str, CacheKeyPrefix], entity_id: Union[int, str], 
                      fetcher: Callable[[], Any], ttl: Optional[int] = None) -> Any:
        """
        Refresh an entity in the cache, fetching it if not found or expired.
        
        Args:
            prefix: Entity type prefix
            entity_id: Entity ID
            fetcher: Function to fetch the entity if not in cache
            ttl: Time-to-live in seconds, or None to use the default TTL
            
        Returns:
            The entity
        """
        key = self.make_key(prefix, str(entity_id))
        value = self.get(key)
        
        if value is None:
            # Not in cache, fetch it
            value = fetcher()
            
            # Cache it if not None
            if value is not None:
                self.set(key, value, ttl)
        
        return value

# Decorators for easy caching

def cached(prefix: Union[str, CacheKeyPrefix], key_fn: Optional[Callable] = None, ttl: Optional[int] = None):
    """
    Decorator for caching function results.
    
    Args:
        prefix: Cache key prefix
        key_fn: Function to generate the cache key from function arguments
        ttl: Time-to-live in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache manager
            cache_manager = CacheManager.get_instance()
            
            # Generate cache key
            if key_fn:
                key_part = key_fn(*args, **kwargs)
            else:
                # Default key is function name + hashable args
                try:
                    arg_str = ":".join(str(arg) for arg in args if arg is not None)
                    kwarg_str = ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()) if v is not None)
                    key_part = f"{func.__name__}:{arg_str}:{kwarg_str}"
                except:
                    # If args are not hashable, use a hash of the function name
                    key_part = f"{func.__name__}:{hash(func)}"
            
            # Create full cache key
            cache_key = cache_manager.make_key(prefix, key_part)
            
            # Try to get from cache
            result = cache_manager.get(cache_key)
            
            if result is None:
                # Not in cache, call the function
                result = func(*args, **kwargs)
                
                # Cache the result if not None
                if result is not None:
                    cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

def cached_property(prefix: Union[str, CacheKeyPrefix], ttl: Optional[int] = None):
    """
    Decorator for caching class property values.
    
    Args:
        prefix: Cache key prefix
        ttl: Time-to-live in seconds
        
    Returns:
        Decorated property
    """
    def decorator(func):
        @property
        @wraps(func)
        def wrapper(self):
            # Get cache manager
            cache_manager = CacheManager.get_instance()
            
            # Generate cache key
            try:
                # Use instance ID or hash
                instance_id = getattr(self, 'id', None) or hash(self)
                cache_key = cache_manager.make_key(prefix, f"{func.__name__}:{instance_id}")
            except:
                # Fallback if instance is not hashable
                cache_key = cache_manager.make_key(prefix, f"{func.__name__}:{id(self)}")
            
            # Try to get from cache
            result = cache_manager.get(cache_key)
            
            if result is None:
                # Not in cache, call the function
                result = func(self)
                
                # Cache the result if not None
                if result is not None:
                    cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# Create singleton instance
cache_manager = CacheManager.get_instance()
```