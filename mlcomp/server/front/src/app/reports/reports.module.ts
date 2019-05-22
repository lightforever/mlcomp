import {NgModule} from '@angular/core';
import {ReportsComponent} from './reports/reports.component'
import {ReportDetailModule} from './report-detail/report-detail.module';
import {ReportRoutingModule} from './report-routing.module'
import {SharedModule} from "../shared.module";
import { ReportsListComponent } from './reports-list/reports-list.component';

@NgModule({
    imports: [
        ReportRoutingModule,
        ReportDetailModule,
        SharedModule
    ],
    declarations: [
        ReportsComponent,
        ReportsListComponent
    ]
})
export class ReportsModule {
}