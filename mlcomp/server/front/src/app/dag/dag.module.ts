import {NgModule} from '@angular/core';
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
    exports: [
        DagComponent
    ],
    declarations: [
        DagComponent
    ]
})
export class DagModule {
}