from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey
from sqlalchemy_serializer import SerializerMixin

Base = declarative_base(cls=SerializerMixin)
