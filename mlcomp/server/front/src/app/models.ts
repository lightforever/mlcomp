export class Project {
  id: number;
  name: string;
  last_activity: Date;
}

export class NameCount {
  name: string;
  count: number;
}
export class Dag {
  id: number;
  name: string;
  created: Date;
  started: Date;
  last_activity: Date;
  finished: Date;
  project: Project;
  task_count: number;
  task_statuses: NameCount;
}

export class Task {
  id: number;
  name: string;
  dag: Dag;
}

export class PaginatorRes<T> {
  data: Array<T>;
  total: number;
}

export class Data<T> {
  data: T
}

export interface FlatNode {
  expandable: boolean;
  name: string;
  level: number;
  content: string;
}

export class CodeNode {
    name: string;
    content: string;
    children?: CodeNode[];
}

export class Computer {
    name: string;
    gpu: number;
    cpu: number;
    memory: number;
}

export class Step {
  id: number;
  task: Task;
  level: number;
  started: Date;
  finished: Date;
  status: string;
  name: string;
}

export class Log {
    id: number;
    message: string;
    time: Date;
    level: string;
    component: string;
    computer: Computer;
    step: Step;
    task: Task;
}

export class Graph {
    nodes: any;
    edges: any;
}

export class PaginatorFilter{
    sort_column: string;
    sort_descending: boolean;
    page_number: number;
    page_size: number;

}

export class LogFilter{
    dag: string;
    task: string;
    components: number[];
    levels: number[];
    step: string;

    task_name: string;
    step_name: string;
    computer: string;

    paginator: PaginatorFilter;

}

export class ProjectFilter {
    paginator: PaginatorFilter;
    name: string;
}

export class DagFilter {
    paginator: PaginatorFilter;
    name: string;
    project: number;
}

export class TaskFilter {
    paginator: PaginatorFilter;
    name: string;
    dag: number;
}