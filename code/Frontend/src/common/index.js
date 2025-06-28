import axios from 'axios';

/**
 * interface Options {
 *  uid: string // 用户账号
 *  channelName: string // 频道名称
 *  appkey: string // 你的 appkey
 *  appSecret: string // 你的 appsecret
 * }
 */
export const getToken = async ({
    uid,
    channelName,
    appkey,
    appSecret
}) => {
    const getTokenUrl = 'https://api.netease.im/nimserver/user/getToken.action';
    const Nonce = Math.ceil(Math.random() * 1e9);
    const CurTime = Math.ceil(Date.now() / 1000);
    const stringToHash = `${appSecret}${Nonce}${CurTime}`;
    
    // 使用浏览器原生加密功能生成 CheckSum
    const encoder = new TextEncoder();
    const data = encoder.encode(stringToHash);
    const hashBuffer = await crypto.subtle.digest('SHA-1', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const CheckSum = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');

    return new Promise((resolve, reject) => {
        axios.post(
            getTokenUrl,
            `uid=${uid}&channelName=${channelName}`, {
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8',
                    AppKey: appkey,
                    Nonce,
                    CurTime,
                    CheckSum,
                }
            }
        ).then(function (response) {
            const data = response.data;
            if (data.code !== 200) {
                reject(new Error('getToken error: ' + data.code));
            } else {
                resolve(data.token);
            }
        }).catch((error) => {
            reject(new Error('getToken error: ' + error));
        });
    });
};

export const checkBrowser = (type) => {
    const ua = navigator.userAgent.toLowerCase();
    const info = {
        ie: /msie/.test(ua) && !/opera/.test(ua),
        opera: /opera/.test(ua),
        safari: /version.*safari/.test(ua),
        chrome: /chrome/.test(ua),
        firefox: /gecko/.test(ua) && !/webkit/.test(ua)
    };
    return info[type] || false;
};
