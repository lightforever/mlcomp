import {NgModule} from '@angular/core';
import {ReportComponent} from './report/report.component'
import {SharedModule} from "../shared.module";
import {ReportRoutingModule} from "./report-routing.module";

@NgModule({
    imports: [
        ReportRoutingModule,
        SharedModule
    ],
    declarations: [
        ReportComponent
    ]
})
export class ReportModule {
}