import {Injectable} from '@angular/core';
import {BaseService} from "./base.service";
import {AppSettings} from "./app-settings";
import {catchError} from "rxjs/operators";
import {BaseResult, DagStopResult, Status, ToogleReportResult} from "./models";

@Injectable({
    providedIn: 'root'
})
export class DagService extends BaseService {
    protected collection_part: string = 'dags';
    protected single_part: string = 'dag';

    stop(id: number) {
        let message = `${this.constructor.name}.stop`;
        return this.http.post<DagStopResult>(AppSettings.API_ENDPOINT + this.single_part + '/stop', {'id': id}).pipe(
            catchError(this.handleError<DagStopResult>(message, new DagStopResult()))
        );
    }

    toogle_report(id: any, report: number, report_full: boolean) {
        let message = `${this.constructor.name}.toogle_report`;
        return this.http.post<ToogleReportResult>(AppSettings.API_ENDPOINT + this.single_part + '/toogle_report', {
            'id': id,
            'report': report,
            'remove': report_full
        }).pipe(
            catchError(this.handleError<ToogleReportResult>(message, new ToogleReportResult()))
        );
    }

    remove(id: number) {
        let message = `${this.constructor.name}.remove`;
        return this.http.post<DagStopResult>(AppSettings.API_ENDPOINT + this.single_part + '/remove', {'id': id}).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    remove_imgs(id: number) {
        let message = `${this.constructor.name}.remove_imgs`;
        return this.http.post<DagStopResult>(AppSettings.API_ENDPOINT + 'remove_imgs', {'dag': id}).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    remove_files(id: number) {
        let message = `${this.constructor.name}.remove_files`;
        return this.http.post<DagStopResult>(AppSettings.API_ENDPOINT + 'remove_files', {'dag': id}).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }
}
