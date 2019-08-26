import time
from math import ceil
from socket import gethostname
import warnings
import datetime
from typing import List

from IPython import display
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.ticker import MaxNLocator

from mlcomp.db.enums import TaskStatus, ComponentType
from mlcomp.db.providers import TaskProvider, LogProvider, \
    DagProvider, ComputerProvider, ReportSeriesProvider
from mlcomp.utils.misc import now, to_snake

warnings.simplefilter('ignore')


def describe_tasks(dag: int, axis):
    provider = TaskProvider()
    columns = ['Name', 'Started', 'Duration', 'Step', 'Status']
    cells = []
    cells_colours = []

    tasks = provider.by_dag(dag)

    status_colors = {
        'not_ran': 'gray',
        'queued': 'lightblue',
        'in_progress': 'lime',
        'failed': '#e83217',
        'stopped': '#cb88ea',
        'skipped': 'orange',
        'success': 'green'
    }

    finish = True

    for task in tasks:
        started = ''
        duration = ''

        if task.status <= TaskStatus.InProgress.value:
            finish = False

        if task.started:
            started = task.started.strftime('%m.%d %H:%M:%S')
            if task.finished:
                duration = (task.finished - task.started).total_seconds()
            else:
                duration = (now() - task.started).total_seconds()

            if duration > 3600:
                duration = f'{int(duration // 3600)} hours ' \
                           f'{int((duration % 3600) // 60)} min' \
                           f' {int(duration % 60)} sec'
            elif duration > 60:
                duration = f'{int(duration // 60)} min' \
                           f' {int(duration % 60)} sec'
            else:
                duration = f'{int(duration)} sec'

        status = to_snake(TaskStatus(task.status).name)
        status_color = status_colors[status]

        task_cells = [task.name, started, duration, task.current_step or '1',
                      status]
        task_colors = ['white', 'white', 'white', 'white', status_color]
        cells.append(task_cells)
        cells_colours.append(task_colors)

    table = axis.table(cellText=cells,
                       colLabels=columns,
                       cellColours=cells_colours,
                       cellLoc='center',
                       colWidths=[0.2, 0.3, 0.4, 0.1, 0.2],
                       bbox=[0, 0, 1.0, 1.0],
                       loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(14)

    axis.set_xticks([])
    axis.axis('off')
    axis.set_title('Tasks')

    return finish


def describe_logs(dag: int, axis, max_log_text: int = None,
                  log_count: int = 5, col_withds: List[float] = None):
    columns = ['Component', 'Level', 'Task', 'Time', 'Text']
    provider = LogProvider()
    logs = provider.last(log_count)

    res = []

    cells = []
    cells_colours = []

    for log, task_name in logs:
        component = to_snake(ComponentType(log.component).name)

        level = log.level
        level = 'debug' if level == 10 else 'info' \
            if level == 20 else 'warning' \
            if level == 30 else 'error'
        message = log.message
        if max_log_text:
            message = message[:max_log_text]
        log_cells = [component, level, task_name,
                     log.time.strftime('%m.%d %H:%M:%S'),
                     message]

        cells.append(log_cells)

        level_color = 'lightblue' if level == 'info' else 'lightyellow' \
            if level == 'warning' else 'red' if level == 'error' else 'white'

        log_colours = ['white', level_color, 'white', 'white', 'white']
        cells_colours.append(log_colours)

        if level == 'error':
            res.append(log)

    col_withds = col_withds or [0.2, 0.1, 0.25, 0.2, 0.45]
    if len(cells) > 0:
        table = axis.table(cellText=cells,
                           colLabels=columns,
                           cellColours=cells_colours,
                           cellLoc='center',
                           colWidths=col_withds,
                           bbox=[0, 0, 1, 1.0],
                           loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(14)

    axis.set_xticks([])
    axis.axis('off')
    axis.set_title('Logs')

    return res


def describe_dag(dag, axis):
    provider = DagProvider()
    graph = provider.graph(dag)

    status_colors = {
        'not_ran': '#808080',
        'queued': '#add8e6',
        'in_progress': '#bfff00',
        'failed': '#e83217',
        'stopped': '#cb88ea',
        'skipped': '#ffa500',
        'success': '#006400'
    }
    node_color = []
    edge_color = []

    G = nx.DiGraph()
    labels = dict()
    for n in graph['nodes']:
        G.add_node(n['id'])
        labels[n['id']] = n['name']
        node_color.append(status_colors[n['status']])

    edges = []
    for e in graph['edges']:
        G.add_edge(e['from'], e['to'])
        edges.append((e['from'], e['to']))
        edge_color.append(status_colors[e['status']])

    pos = nx.spring_layout(G, seed=0)
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_color,
                           ax=axis,
                           node_size=2000)
    nx.draw_networkx_labels(G, pos, labels, ax=axis, with_labels=True,
                            font_color='orange',
                            font_weight='bold',
                            font_size=18)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_color,
                           arrows=True, arrowsize=80, ax=axis)

    axis.set_xticks([])
    axis.axis('off')
    axis.set_title('Graph')


def describe_resources(computer: str, axis):
    provider = ComputerProvider()
    res = provider.get({})['data']
    res = [r for r in res if r['name'] == computer][0]
    usage = res['usage_history']
    x = [datetime.datetime.strptime(t, provider.datetime_format) for t in
         usage['time']]

    for item in usage['mean']:
        if item['name'] == 'disk':
            continue

        axis.plot(x, item['value'], label=item['name'])

    axis.set_title('Resources')
    axis.set_ylabel('%')
    axis.legend(loc='lower left')


def describe_metrics(dag: int, metrics: List[str], axis, last_n_epoch=None):
    metrics = metrics or []

    series_provider = ReportSeriesProvider()
    series = series_provider.by_dag(dag, metrics)

    for i in range(len(axis)):
        ax = axis[i]
        if i >= len(series):
            ax.axis('off')
            continue

        ax.axis('on')
        task_name, metric, groups = series[i]

        for group in groups:
            if last_n_epoch:
                group['epoch'] = group['epoch'][-last_n_epoch:]
                group['value'] = group['value'][-last_n_epoch:]

            ax.plot(group['epoch'], group['value'], label=group['name'])

        ax.set_title(f'{task_name}, {metric} score')
        ax.set_ylabel(metric, labelpad=20)
        ax.set_xlabel('epoch')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()


def describe(dag: int, metrics=None, last_n_epoch=None,
             computer: str = None, max_log_text: int = 45,
             fig_size=(12, 10), grid_spec: dict = None,
             log_count=5, log_col_widths: List[float] = None,
             wait=True, wait_interval=5):
    grid_spec = grid_spec or {}
    metrics = metrics or []
    size = (4 + ceil(len(metrics) / 2), 2)
    default_grid_spec = {
        'tasks': {'rowspan': 1, 'colspan': 2, 'loc': (0, 0)},
        'dag': {'rowspan': 1, 'colspan': 2, 'loc': (1, 0)},
        'logs': {'rowspan': 1, 'colspan': 2, 'loc': (2, 0)},
        'resources': {'rowspan': 1, 'colspan': 2, 'loc': (3, 0)},
        'size': size
    }
    loc = (4, 0)
    for m in metrics:
        default_grid_spec[m] = {'rowspan': 1, 'colspan': 1, 'loc': loc}
        if loc[1] == 1:
            loc = (loc[0] + 1, 0)
        else:
            loc = (loc[0], 1)

    default_grid_spec.update(grid_spec)
    grid_spec = default_grid_spec

    fig = plt.figure(figsize=fig_size)

    def grid_cell(spec: dict):
        return plt.subplot2grid(size, spec['loc'],
                                colspan=spec['colspan'],
                                rowspan=spec['rowspan'],
                                fig=fig
                                )

    while True:
        computer = computer or gethostname()

        task_axis = grid_cell(grid_spec['tasks'])
        dag_axis = grid_cell(grid_spec['dag'])
        resources_axis = grid_cell(grid_spec['resources'])
        logs_axis = grid_cell(grid_spec['logs'])

        finish = describe_tasks(dag, task_axis)
        describe_dag(dag, dag_axis)
        errors = describe_logs(dag, axis=logs_axis,
                               max_log_text=max_log_text,
                               log_count=log_count,
                               col_withds=log_col_widths)
        describe_resources(computer=computer, axis=resources_axis)

        metric_axis = [grid_cell(grid_spec[m]) for m in metrics]

        describe_metrics(dag, metrics,
                         last_n_epoch=last_n_epoch,
                         axis=metric_axis
                         )

        plt.tight_layout()

        display.clear_output(wait=True)

        for error in errors:
            print(error.time)
            print(error.message)

        display.display(fig)

        if not wait or finish:
            break

        time.sleep(wait_interval)

    plt.close(fig)


__all__ = ['describe']

if __name__ == '__main__':
    describe(dag=293, metrics=['loss', 'dice'], wait_interval=2)
