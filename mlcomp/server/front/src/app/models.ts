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

export class Graph {
    nodes: any;
    edges: any;
}