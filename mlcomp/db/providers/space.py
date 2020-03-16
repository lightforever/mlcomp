from sqlalchemy import literal_column, func

from mlcomp.db.core import PaginatorOptions
from mlcomp.db.models import Space
from mlcomp.db.models.space import SpaceRelation, SpaceTag
from mlcomp.db.providers.base import BaseDataProvider


class SpaceProvider(BaseDataProvider):
    model = Space

    def get(self, filter: dict, options: PaginatorOptions = None):
        query = self.query(Space, literal_column('0').label('relation'))
        if 'parent' in filter:
            query = query.filter(Space.name != filter['parent'])

        if filter.get('name'):
            query = query.filter(Space.name.contains(filter['name']))

        tags = filter.get('tags', [])
        if len(tags) > 0:
            query = query.join(SpaceTag).filter(SpaceTag.tag.in_(tags))

        if filter.get('parent'):
            relation = literal_column('1').label('relation')
            query2 = self.query(Space, relation). \
                join(SpaceRelation, SpaceRelation.child == Space.name).filter(
                SpaceRelation.parent == filter['parent']
            )
            query2_names = [space.name for space, _ in query2.all()]
            query = query.filter(Space.name.notin_(query2_names))
            query = query.union_all(query2)
            query = query.order_by(relation.desc())

        total = query.count()
        paginator = self.paginator(query, options) if options else query
        data = []

        for space, relation in paginator.all():
            item = self.to_dict(space)
            item['relation'] = relation
            data.append(item)

        names = [d['name'] for d in data]
        tags = self.query(SpaceTag).filter(SpaceTag.space.in_(names))
        for d in data:
            d['tags'] = []
            for t in tags:
                if d['name'] == t.space:
                    d['tags'].append(t.tag)

        return {
            'total': total,
            'data': data
        }

    def tags(self, name: str):
        tag_count = func.count(SpaceTag.tag).label('count')
        query = self.query(SpaceTag.tag, tag_count)
        if name:
            query = query.filter(SpaceTag.tag.contains(name))

        tags = query.group_by(
            SpaceTag.tag).order_by(tag_count.desc()).limit(10)
        tags = [t for t, c in tags]
        return {'tags': tags}

    def add_relation(self, parent: str, child: str):
        self.add(SpaceRelation(parent=parent, child=child))

    def remove_relation(self, parent: str, child: str):
        self.query(SpaceRelation).filter(
            SpaceRelation.parent == parent).filter(
            SpaceRelation.child == child).delete(synchronize_session=False)

        self.session.commit()

    def related(self, parent: str):
        res = self.query(Space).join(SpaceRelation,
                                     SpaceRelation.child == Space.name) \
            .filter(SpaceRelation.parent == parent) \
            .all()
        return res

    def remove_tag(self, space: str, tag: str):
        self.query(SpaceTag).filter(SpaceTag.space == space).filter(
            SpaceTag.tag == tag).delete(synchronize_session=False)

        self.session.commit()

    def names(self, name: str):
        query = self.query(Space.name)
        if name:
            query = query.filter(Space.name.contains(name))

        res = query.group_by(Space.name).order_by(Space.name).limit(10).all()
        res = [r[0] for r in res]
        return {'names': res}


__all__ = ['SpaceProvider']
