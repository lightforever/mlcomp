import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';

import {TaskDetailComponent} from "./task-detail/task-detail.component";
import {LogComponent} from "../../log/log.component";


const routes: Routes = [
    {

        path: '',
        component: TaskDetailComponent,
        children: [
            {path: '', component: LogComponent}
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
export class TaskDetailRoutingModule {
}