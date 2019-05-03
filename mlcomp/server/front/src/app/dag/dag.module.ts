import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';
import {DagsComponent} from './dags/dags.component'
import {DemoMaterialModule} from '../material-module';
// import {DagDetailModule} from './dag-detail/dag-detail.module';

import { DagComponent } from './dag/dag.component';
import {DagRoutingModule} from './dag-routing.module'

@NgModule({
    imports: [
        CommonModule,
        DemoMaterialModule,
        DagRoutingModule,
        // DagDetailModule
    ],
    declarations: [
        DagsComponent,
        DagComponent
    ]
})
export class DagModule {
}