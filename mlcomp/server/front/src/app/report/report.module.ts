import {NgModule} from '@angular/core';
import {ReportRoutingModule} from "./report-routing.module";
import {SharedModule} from "../shared.module";
import {SeriesComponent} from './internal/series/series.component';
import {ImgComponent} from './internal/img/img.component';
import {ImgClassifyComponent} from "./internal/img-classify/img-classify.component";
import {ReportComponent} from "./report/report.component";
import {LayoutComponent} from "./internal/layout/layout.component";
import {ReportDetailComponent} from "./report-detail/report-detail.component";
import { LayoutsComponent } from './layouts/layouts.component';
import {TableComponent} from "./internal/table/table.component";
import {ImgSegmentComponent} from "./internal/img-segment/img-segment.component";

@NgModule({
    imports: [
        ReportRoutingModule,
        SharedModule
    ],
    declarations: [
        SeriesComponent,
        ImgComponent,
        ImgClassifyComponent,
        ImgSegmentComponent,
        ReportComponent,
        LayoutComponent,
        ReportDetailComponent,
        LayoutsComponent,
        TableComponent
    ]
})
export class ReportModule {
}