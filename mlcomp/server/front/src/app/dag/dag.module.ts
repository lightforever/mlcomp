import {NgModule} from '@angular/core';
import {DagsComponent} from './dags/dags.component'
import {DagDetailModule} from './dag-detail/dag-detail.module';
import { DagComponent } from './dag/dag.component';
import {DagRoutingModule} from './dag-routing.module'
import {SharedModule} from "../shared.module";

@NgModule({
    imports: [
        DagRoutingModule,
        DagDetailModule,
        SharedModule
    ],
    declarations: [
        DagsComponent,
        DagComponent
    ]
})
export class DagModule {
}