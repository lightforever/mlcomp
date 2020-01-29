Run

```bash
mlcomp dag examples/grid/dag.yml
mlcomp dag examples/grid/task.yml
```

Both dag and task can apply grids.

Grid applied to DAG: there are multiple dags are being created.

Grid applied to Task: the dag is the same. There will be many instances of this task. 