import {NgModule} from '@angular/core';
import {TaskDetailModule} from './task-detail/task-detail.module';
import {TaskRoutingModule} from './task-routing.module'
import {SharedModule} from "../shared.module";

@NgModule({
    imports: [
        TaskRoutingModule,
        TaskDetailModule,
        SharedModule
    ],
    declarations: [
    ]
})
export class TaskModule {
}