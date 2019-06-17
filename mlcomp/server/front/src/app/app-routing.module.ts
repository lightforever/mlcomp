import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';

import {ProjectComponent} from './project/project.component';
import {ComputerComponent} from './computer/computer.component';
import {LogComponent} from "./log/log.component";
import { AuthGuard } from './_helpers/auth.gaurd';
import {LoginComponent} from "./login/login.component";
import {ModelComponent} from "./model/model.component";

const routes: Routes = [
    {path: '', redirectTo: '/projects', pathMatch: 'full', canActivate: [AuthGuard] },
    { path: 'login', component: LoginComponent },
    {path: 'projects', component: ProjectComponent, canActivate: [AuthGuard] },
    {path: 'computers', component: ComputerComponent, canActivate: [AuthGuard] },
    {path: 'tasks', loadChildren: './task/task.module#TaskModule', canActivate: [AuthGuard] },
    {path: 'models', component: ModelComponent, canActivate: [AuthGuard] },
    {path: 'dags', loadChildren: './dag/dag.module#DagModule', canActivate: [AuthGuard] },
    {path: 'logs', component: LogComponent, canActivate: [AuthGuard] },
    {path: 'reports', loadChildren: './report/report.module#ReportModule', canActivate: [AuthGuard] },
];

@NgModule({
    imports: [RouterModule.forRoot(routes)],
    exports: [RouterModule]
})
export class AppRoutingModule {
}
