import {Injectable} from '@angular/core';
import {BaseService} from "../base.service";
import {AppSettings} from "../app-settings";
import {catchError} from "rxjs/operators";
import {
    BaseResult,
    DagRestart,
    DagStopResult, TagResult,
    ToogleReportResult
} from "../models";

@Injectable({
    providedIn: 'root'
})
export class DagService extends BaseService {
    protected collection_part: string = 'dags';
    protected single_part: string = 'dag';

    stop(id: number) {
        let message = `${this.constructor.name}.stop`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/stop';
        return this.http.post<DagStopResult>(url, {'id': id}).pipe(
            catchError(this.handleError<DagStopResult>(message,
                new DagStopResult()))
        );
    }

    toogle_report(id: any, report: number, report_full: boolean) {
        let message = `${this.constructor.name}.toogle_report`;
        let url = AppSettings.API_ENDPOINT +
            this.single_part +
            '/toogle_report';

        return this.http.post<ToogleReportResult>(url, {
            'id': id,
            'report': report,
            'remove': report_full
        }).pipe(
            catchError(this.handleError<ToogleReportResult>(message,
                new ToogleReportResult()))
        );
    }

    remove(id: number) {
        let message = `${this.constructor.name}.remove`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/remove';
        return this.http.post<DagStopResult>(url, {'id': id}).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    remove_imgs(id: number) {
        let message = `${this.constructor.name}.remove_imgs`;
        let url = AppSettings.API_ENDPOINT + 'remove_imgs';
        return this.http.post<DagStopResult>(url, {'dag': id}).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    remove_files(id: number) {
        let message = `${this.constructor.name}.remove_files`;
        let url = AppSettings.API_ENDPOINT + 'remove_files';
        return this.http.post<DagStopResult>(url, {'dag': id}).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    start(id: number) {
        let message = `${this.constructor.name}.start`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/start';
        return this.http.post<DagStopResult>(url, {'id': id}).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    restart(data: DagRestart) {
        let message = `${this.constructor.name}.restart`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/restart';
        return this.http.post<BaseResult>(url, data).pipe(
            catchError(this.handleError<BaseResult>(message,
                new BaseResult()))
        );
    }

    tag_add(dag: number, tag: string) {
        let message = `${this.constructor.name}.tag_add`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/tag_add';
        return this.http.post<DagStopResult>(url, {
            'dag': dag,
            'tag': tag
        }).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    tag_remove(dag: number, tag: string) {
        let message = `${this.constructor.name}.tag_remove`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/tag_remove';
        return this.http.post<DagStopResult>(url, {
            'dag': dag,
            'tag': tag
        }).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    tags(data) {
        let message = `${this.constructor.name}.tags`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/tags';
        return this.http.post<TagResult>(url, data).pipe(
            catchError(this.handleError<TagResult>(message, new TagResult()))
        );
    }

}
