from glob import glob
import os
import pickle
from mlcomp.db.providers import ReportSchemeProvider, ReportScheme
from mlcomp.utils.misc import now


def upgrade(migrate_engine):
    folder = os.path.dirname(__file__)
    provider = ReportSchemeProvider()
    try:
        files = os.path.join(folder, '002', 'report_scheme', '*.yml')
        for path in glob(files):
            name = os.path.basename(path).split('.')[0]
            text = open(path).read()
            provider.add(ReportScheme(
                name=name,
                content=pickle.dumps(text),
                last_modified=now()),
                commit=False)

        provider.commit()
    except:
        provider.rollback()
        raise


def downgrade(migrate_engine):
    provider = ReportSchemeProvider()
    provider.session.query(ReportScheme).delete(synchronize_session=False)
    provider.session.commit()
