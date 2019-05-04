import {NgModule} from '@angular/core';
import {TasksComponent} from './tasks/tasks.component'
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
        TasksComponent
    ]
})
export class TaskModule {
}