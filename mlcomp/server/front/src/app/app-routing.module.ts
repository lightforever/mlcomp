import { NgModule }             from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

import { ProjectComponent }   from './project/project.component';
import { DagComponent }   from './dag/dag.component';
import { DagDetailComponent }   from './dag-detail/dag-detail.component';
import { ComputerComponent }   from './computer/computer.component';
import { TaskComponent }      from './task/task.component';
import { TaskDetailComponent }  from './task-detail/task-detail.component';

const routes: Routes = [
  { path: '', redirectTo: '/projects', pathMatch: 'full' },
  { path: 'projects', component: ProjectComponent },
  { path: 'computers', component: ComputerComponent },
  { path: 'dags', component: DagComponent },
  { path: 'dag/:id', component: DagDetailComponent },
  { path: 'tasks/:id', component: TaskDetailComponent },
  { path: 'tasks', component: TaskComponent }
];

@NgModule({
  imports: [ RouterModule.forRoot(routes) ],
  exports: [ RouterModule ]
})
export class AppRoutingModule {}
