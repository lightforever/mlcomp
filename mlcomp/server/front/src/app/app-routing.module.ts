import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';

import {ProjectComponent} from './project/project.component';
import {ComputerComponent} from './computer/computer.component';
import {TaskComponent} from './task/task.component';
import {TaskDetailComponent} from './task-detail/task-detail.component';
import {LogComponent} from "./log/log.component";
import {ReportComponent} from "./report/report.component";

const routes: Routes = [
    {path: '', redirectTo: '/projects', pathMatch: 'full'},
    {path: 'projects', component: ProjectComponent},
    {path: 'computers', component: ComputerComponent},
    {path: 'tasks/:id', component: TaskDetailComponent},
    {path: 'tasks', component: TaskComponent},
    {path: 'dags', loadChildren: './dag/dag.module#DagModule'},
    {path: 'logs', component: LogComponent},
    {path: 'reports', component: ReportComponent},
];

@NgModule({
    imports: [RouterModule.forRoot(routes)],
    exports: [RouterModule]
})
export class AppRoutingModule {
}
