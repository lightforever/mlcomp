import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';
import {ReportsComponent} from "./reports/reports.component";

const routes: Routes = [
    {
        path: '',
        component: ReportsComponent,
        // children: [
        //         {path: 'task-detail/:id', loadChildren: './task-detail/task-detail.module#TaskDetailModule'},
        //          {path: '', component: TasksComponent}
        // ]
    }
];

@NgModule({
    imports: [
        RouterModule.forChild(routes)
    ],
    exports: [
        RouterModule
    ]
})
export class ReportsRoutingModule {
}
