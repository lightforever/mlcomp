from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_serializer import SerializerMixin

Base = declarative_base(cls=SerializerMixin)
