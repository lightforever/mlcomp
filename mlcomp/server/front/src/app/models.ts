export class Project {
  id: number;
  name: string;
  tasks: Task[];
  lastActivity: Date;
}

export class Task {
  id: number;
  name: string;
  project: Project;
}
