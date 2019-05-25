import {NgModule} from '@angular/core';
import {CodeComponent} from './code/code.component';
import {ConfigComponent} from './config/config.component';
import {GraphComponent} from './graph/graph.component';

import {DagDetailRoutingModule} from './dag-detail-routing.module';
import { DagDetailComponent } from './dag-detail/dag-detail.component';
import {SharedModule} from "../../shared.module";
import {TasksHostComponent} from "./tasks/tasks.component";

@NgModule({
    imports: [
        DagDetailRoutingModule,
        SharedModule
    ],
    declarations: [
        CodeComponent,
        ConfigComponent,
        GraphComponent,
        DagDetailComponent,
        TasksHostComponent
    ]
})
export class DagDetailModule {
}