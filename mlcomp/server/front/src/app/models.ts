export class BaseResult {
    success: boolean;
    error: string;
}

export class UpdateCodeResult extends BaseResult{
    file: number;
}

export class Project {
  id: number;
  name: string;
  last_activity: Date;
  img_size: number;
  file_size: number;
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
  img_size: number;
  file_size: number;
  report: number;
}

export class Task {
  id: number;
  name: string;
  dag: Dag;
  type: string;
  report: number;
}

export class PaginatorRes<T> {
  data: Array<T> = [];
  total: number = 0;
}

export class Data<T> {
  data: T
}

export interface FlatNode {
  expandable: boolean;
  name: string;
  level: number;
  content: any;
  id: number;
  dag: number;
  storage: number;
}

export class CodeNode {
    name: string;
    content: string;
    id: number;
    children?: CodeNode[];
    dag: number;
    storage: number;
}

export class CodeResult {
    items: CodeNode[];
}

export class Gpu {
    memory: number;
    load: number;
}

export class Usage {
    cpu: number;
    gpu: Gpu[];
    memory: number;
}

export class ComputerUsage {
    mean: Usage[];
    peak: Usage[];
}

export class Computer {
    name: string;
    gpu: number;
    cpu: number;
    memory: number;
    usage: Usage;
    dockers: any[];
    sync_status: string;
    sync_date: Date;
}

export class SyncProject {
    id: number;
    name: string;
    ignore_folders: string;
}

export class SyncStart extends BaseResult {
    projects: SyncProject[];
    project: SyncProject;
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
    message: string[];
    time: Date;
    level: string;
    component: string;
    computer: Computer;
    step: Step;
    task: Task;
    module: string;
    line: number;
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

export class ComputerFilter {
    paginator: PaginatorFilter;
    usage_min_time: Date;
}

export class LogFilter{
    dag: number;
    task: number;
    components: number[];
    levels: number[];
    step: number;

    task_name: string;
    step_name: string;
    computer: string;
    message: string;

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
    report: number;
    id: number;
    status;
    created_min: Date;
    created_max: Date;
    last_activity_min: Date;
    last_activity_max: Date;
    id_min: number;
    id_max: number;
}

export class TaskFilter {
    paginator: PaginatorFilter;
    name: string;
    dag: number;
    report: number;
    id: number;

    status;
    created_min: Date;
    created_max: Date;
    last_activity_min: Date;
    last_activity_max: Date;
    project: number;
    type: string[];
    id_min: number;
    id_max: number;
}

export class ReportsFilter {
    paginator: PaginatorFilter;
    task: number;
    dag: number;
}

export class ToogleReportResult extends BaseResult{
    report_full: boolean;
}

export class Status extends BaseResult{
    status: string;
}

export class DagStopResult extends BaseResult{
    dag: Dag;
}

export class ReportItem {
    type: string;
    items: ReportItem[];
    title: string;
    parent_cols: number;
    cols: number;
    rows: number;
    row_height: number;
    source;
    expanded: boolean;
    multi: boolean;
    group: string[];
    table: boolean;
    name: string;
    data: any;
    part: string;
    index: number;
    attrs: ImgAttr[];
}

export class Metric{
    name: string;
    minimize: boolean;
}

export class Report {
    layout;
    data;
    metric: Metric;
}

export class StepNode {
    name: string;
    id: number;
    children?: StepNode[];
}

export class StepNodeResult {
    data: StepNode[];
}

export class Img {
    content: string;
}

export class Series {
    x: number[];
    y: number[];
    time: string[];
    text: string[];
    stage: string[];
    color: string;
    name: string;
    group: string;
    task_name: string;
    task_id: number;
    source: string;
    plotted: number;
}

export class SeriesItem {
    name: string;
    series: Series[];
    layout: any;
}

export class ImgAttr {
    type: string;
    source: string;
    name: string;
    equal: any;
    greater: number;
    less: number;
}

export class ImgClassify {
    name: string;
    task: number;
    group: string;
}

export class Model {
    id: number;
    name: string;
    score_local: number;
    score_public: number;
    task: number;
    project: number;
    created: Date;
    dag_rel: Dag;
}

export class ModelFilter {
    paginator: PaginatorFilter;
    name: string;
    project: number;
    created_min: Date;
    created_max: Date;
}

export interface ProjectAddData {
    name: string;
    class_names: string;
    ignore_folders: string;
}

export interface ReportAddData {
    name: string;
    project: number;
    layout: string;
    projects: Project[];
    layouts: any[];
}

export class ReportUpdateData {
    id: number;
    layout: string;
    layouts: string[];
}

export interface ModelAddData {
    project: number;
    equations: string;
    projects: any[];
    name: string;
    task: number;
    file: string;
    fold: number;
}

export class ModelStartData extends BaseResult{
    dags: any[];
    dag: any;
    pipe: any;
    model_id: number;
}

export class Layout {
    name: string;
    content: string;
    last_modified: Date;
}

export interface LayoutAddData {
    name: string;
}

export class AuxiliarySupervisorComputer {
    name: string;
    expanded: boolean;
    cpu: number;
    cpu_total: number;
    memory: number;
    memory_total: number;
    gpu: number[];
    ports: number[];
}

export class AuxiliarySupervisorParentTask {
    name: string;
    id: number;
    started: Date;
    statuses;
    expanded: boolean;
}

export class AuxiliarySupervisorNotRanTask {
    id: number;
    name: string;
    dep_status: string[];
}

export class AuxiliarySupervisorProcessTask {
    not_valid: string;
    computers: any;
    to_send: any;
    id: number;
    name: string;
}

export class AuxiliarySupervisor {
    queues: string[];
    time: Date;
    duration: number;
    computers: AuxiliarySupervisorComputer[];
    parent_tasks_stats: AuxiliarySupervisorParentTask[];
    not_ran_tasks: AuxiliarySupervisorNotRanTask[];
    process_tasks: AuxiliarySupervisorProcessTask[];
}

export class Auxiliary {
    supervisor: AuxiliarySupervisor;
}

export class TaskInfo {
    id: number;
    pid: number;
    worker_index: number;
    gpu_assigned: number;
    celery_id: string;
    additional_info: string;
    result: string;
}