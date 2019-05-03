import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';

import {CodeComponent} from './code/code.component';
import {ConfigComponent} from './config/config.component';
import {GraphComponent} from './graph/graph.component';

import {DagDetailRoutingModule} from './dag-detail-routing.module';
import { DagDetailComponent } from './dag-detail/dag-detail.component';
import {CdkTreeModule} from '@angular/cdk/tree';
import {MatTreeModule, MatIconModule} from '@angular/material'

@NgModule({
    imports: [
        CommonModule,
        DagDetailRoutingModule,
        CdkTreeModule,
        MatTreeModule,
        MatIconModule
    ],
    declarations: [
        CodeComponent,
        ConfigComponent,
        GraphComponent,
        DagDetailComponent
    ]
})
export class DagDetailModule {
}