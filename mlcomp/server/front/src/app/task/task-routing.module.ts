import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';
import {TasksComponent} from "./tasks/tasks.component";
import {TaskComponent} from "./task/task.component";

const routes: Routes = [
    {
        path: '',
        component: TaskComponent,
        children: [
                {path: 'task-detail/:id', loadChildren: './task-detail/task-detail.module#TaskDetailModule'},
                 {path: '', component: TasksComponent}
        ]
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
export class TaskRoutingModule {
}


/*
Copyright Google LLC. All Rights Reserved.
Use of this source code is governed by an MIT-style license that
can be found in the LICENSE file at http://angular.io/license
*/