export class Project {
  id: number;
  name: string;
  tasks: Task[];
  last_activity: Date;
}

export class Task {
  id: number;
  name: string;
  project: Project;
}
