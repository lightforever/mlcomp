import {NgModule} from '@angular/core';
import {ReportRoutingModule} from "./report-routing.module";
import {SharedModule} from "../shared.module";

@NgModule({
    imports: [
        ReportRoutingModule,
        SharedModule
    ],
    exports: [

    ],
    declarations: []
})
export class ReportModule {
}