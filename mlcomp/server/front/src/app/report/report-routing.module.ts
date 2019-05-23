import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';
import {ReportComponent} from "./report/report.component";
import {ReportsComponent} from "./reports/reports.component";
import {ReportDetailComponent} from "./report-detail/report-detail/report-detail.component";

const routes: Routes = [
    {
        path: '',
        component: ReportComponent,
        children: [
                {path: '', component: ReportsComponent},
                {path: 'report-detail/:id', component: ReportDetailComponent},
        ]
    }
];

@NgModule({
    imports: [
        RouterModule.forChild(routes)
    ],
    exports: [
        RouterModule
    ]
})
export class ReportRoutingModule {
}
