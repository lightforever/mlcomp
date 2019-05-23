import {NgModule} from '@angular/core';
import {ReportComponent} from './report/report.component'
import {ReportsComponent} from './reports/reports.component';
import {SharedModule} from "../shared.module";
import {ReportRoutingModule} from "./report-routing.module";

@NgModule({
    imports: [
        ReportRoutingModule,
        SharedModule
    ],
    declarations: [
        ReportComponent,
        ReportsComponent
    ]
})
export class ReportModule {
}