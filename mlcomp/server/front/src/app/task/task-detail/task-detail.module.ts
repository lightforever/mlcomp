import {NgModule} from '@angular/core';

import {TaskDetailRoutingModule} from './task-detail-routing.module';
import {SharedModule} from "../../shared.module";
import {TaskDetailComponent} from "./task-detail/task-detail.component";

@NgModule({
    imports: [
        TaskDetailRoutingModule,
        SharedModule
    ],
    declarations: [
        TaskDetailComponent
    ]
})
export class TaskDetailModule {
}