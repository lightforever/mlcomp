import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';

import {CodeComponent} from './code/code.component';
import {ConfigComponent} from './config/config.component';
import {TasksComponent} from './tasks/tasks.component';
import {GraphComponent} from './graph/graph.component';

import {DagDetailRoutingModule} from './dag-detail-routing.module';
import { DagDetailComponent } from './dag-detail/dag-detail.component';

@NgModule({
    imports: [
        CommonModule,
        DagDetailRoutingModule,
    ],
    declarations: [
        CodeComponent,
        ConfigComponent,
        TasksComponent,
        GraphComponent,
        DagDetailComponent
    ]
})
export class DagDetailModule {
}